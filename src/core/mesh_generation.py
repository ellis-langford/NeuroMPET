# Imports
import os
import shutil
from datetime import datetime
import simpleware.scripting as sw
from simpleware.scripting import SurfaceFixingControlParameters, Model

# --------------- CONFIG & INPUTS ----------------
BASE_DIR = r"D:\ellis\mnt\projects\neuro_mpet\workflow_processing"
SURFACE_DIR = os.path.join(BASE_DIR, "3.surface_generation")
OUTDIR = os.path.join(BASE_DIR, "4.meshing")
# SUBJECTS = ["OAS30026_MR_d0129"]
SUBJECTS = os.listdir(os.path.join(BASE_DIR, "data"))

# Target and tolerances (configurable)
TARGET_GLOBAL_ELEMENTS = 2_500_000            # Target for global mesh elements
TOLERANCE_FRAC = 0.20                         # Element count relative tolerance (20%)
COARSENESS_STEPS = 15                         # Number of mesh_coarseness values to try

REGIONS = {
    "global": -50,
    "brainstem_L": -50,
    "brainstem_R": -50,
    "cerebrum_L": -50,
    "cerebrum_R": -50,
    "cerebrumWM_L": -50,
    "cerebrumWM_R": -50,
    "cerebellum_L": -50,
    "cerebellum_R": -50,
    "cerebellumWM_L": -50,
    "cerebellumWM_R": -50
}

# ------------------ FUNCTIONS -------------------
def get_parameters(surface_path, volume_fpath=""):
    doc = sw.App.GetDocument()
    doc.ImportSurfaceFromStlFile(surface_path, False, 1.0, False)
    num_triangles = doc.GetActiveSurface().GetPolygonCount()
    b = doc.GetActiveSurface().GetBounds()
    bbox_volume = (b[1]-b[0])*(b[3]-b[2])*(b[5]-b[4])
    doc.RemoveSurface(doc.GetActiveSurface())

    if volume_fpath:
        doc.ImportVolumeMeshFromFile(volume_fpath, 1.0, False)
        num_elements = doc.GetActiveVolumeMesh().GetElementCount()
        doc.RemoveVolumeMesh(doc.GetActiveVolumeMesh())
        return bbox_volume, num_triangles, num_elements
    else:
        return bbox_volume, num_triangles

def fix_surface(subject, region, start_coarse, surface_path, outdir, rerun=False):
    """
    Try to import surface, fix it (1 or 2 rounds), create FE model, generate mesh, export vtk.
    Returns: True if surface successfully fixed.
    If mesh generation fails returns False and writes errors.txt in outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    doc = sw.App.GetDocument()

    doc.ImportSurfaceFromStlFile(surface_path, False, 1.0, False)
    surface = doc.GetActiveSurface()
    surface.SetName(region)

    # Apply surface fixing
    surface.Fix(SurfaceFixingControlParameters(9.9999999999999995e-07,
                                                1.0000000000000001e-09))

    if rerun:
        surface.Fix(SurfaceFixingControlParameters(9.9999999999999995e-07,
                                                    1.0000000000000001e-09))

    # Triangle count & basic sanity check
    num_triangles = surface.GetPolygonCount()
    if num_triangles < 500:
        # Too small number of triangles to be valid surface
        with open(os.path.join(outdir, "errors.txt"), "a") as f:
            now = datetime.now().strftime('%d-%m-%Y %H:%M')
            f.write(f"[ Error | {now} ] Fixed surface does not contain enough elements {subject} {region}\n")
        doc.RemoveSurface(surface)
        success = False
        return success

    # Setup for meshing
    doc.CreateFeModel(region)
    doc.EnableObjectsMode()
    model = doc.GetModelByName(region)
    model.AddSurface(surface)
    doc.EnableModelsMode()
    model.SetExportType(Model.VtkVolume)
    model.SetCompoundCoarsenessOnPart(model.GetPartByName(region), start_coarse)

    # Attempt to generate mesh
    try:
        doc.GenerateMesh()
        outpath = os.path.join(outdir, f"{region}.vtk")
        doc.ExportVtkVolume(outpath, False)
        success = True
        surface.Export(os.path.join(outdir, f"{region}.stl"), False)
        doc.RemoveModel(model)
        doc.RemoveSurface(surface)
    except Exception as e:
        success = False
        doc.RemoveSurface(surface)
        return success
    
    return success

def generate_mesh(subject, region, coarseness, surface_path, outdir, target_elements):
    """
    Try to import surface, fix it (1 or 2 rounds), create FE model, generate mesh, export vtk.
    Returns: (num_elements, bbox_volume, mesh_generated)
    If mesh generation fails returns (None, None, False) and writes errors.txt in outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    doc = sw.App.GetDocument()

    # Import STL
    doc.ImportSurfaceFromStlFile(surface_path, False, 1.0, False)
    surface = doc.GetActiveSurface()
    surface.SetName(region)

    # Get surface parameters    
    num_triangles = surface.GetPolygonCount()
    b = surface.GetBounds()
    bbox_volume = (b[1]-b[0])*(b[3]-b[2])*(b[5]-b[4])

    # Create FE model and add surface
    doc.CreateFeModel(region)
    doc.EnableObjectsMode()
    model = doc.GetModelByName(region)
    model.AddSurface(surface)

    # Set model export type and coarseness
    doc.EnableModelsMode()
    model.SetExportType(Model.VtkVolume)
    model.SetCompoundCoarsenessOnPart(model.GetPartByName(region), coarseness)

    # Attempt to generate mesh
    try:
        doc.GenerateMesh()
    except Exception as e:
        mesh_generated = False
        doc.RemoveModel(model)
        doc.RemoveSurface(surface)

        return None, None, mesh_generated

    # If mesh generation succeeded, export VTK volume
    outpath = os.path.join(outdir, f"{region}.vtk")
    doc.ExportVtkVolume(outpath, False)

    # Import the exported mesh to query element counts (then remove)
    doc.ImportVolumeMeshFromFile(outpath, 1.0, False)
    vm = doc.GetActiveVolumeMesh()
    num_elements = vm.GetElementCount()
    
    # Write results file with target and percentage difference (if available)
    results_txt_path = os.path.join(outdir, "results.txt")
    with open(results_txt_path, "a") as f:
        pct_diff = 100 * (num_elements / target_elements)
        f.write(f"Subject {subject} Region {region} Coarseness {coarseness}\n")
        f.write(f"BoundingBoxSize {bbox_volume:.0f}\n")
        f.write(f"SurfaceElements {num_triangles}\n")
        f.write(f"VolumeElements {num_elements}\n")
        f.write(f"TargetElements {target_elements:.0f}\n")
        f.write(f"PercentDiff {pct_diff:.0f}%\n\n")

    # Clean up model and surface
    doc.RemoveModel(model)
    doc.RemoveSurface(surface)
    doc.RemoveVolumeMesh(vm)
    mesh_generated = True

    return num_elements, bbox_volume, mesh_generated

# ------------------ PROCESSING --------------------
def main():
    for subject in SUBJECTS:
        subject_outdir = os.path.join(OUTDIR, subject)
        os.makedirs(subject_outdir, exist_ok=True)

        # Record start of meshing
        with open(os.path.join(subject_outdir, "results.txt"), "w") as rf:
            rf.write(f"All meshes produced successfully {subject}\n")
            now = datetime.now().strftime('%d-%m-%Y %H:%M')
            rf.write(f"[ Log | {now} ] Starting meshing\n")

        # -------- FIX WHOLEBRAIN & VENTRICLE SURFACES --------
        for region in ["wholebrain", "ventricles"]:
            surface_path = os.path.join(SURFACE_DIR,  subject, "outputs", "surfaces", f"{region}.stl")
            outdir_region = os.path.join(subject_outdir, region)
            surface_fixed = fix_surface(subject, region, -50, surface_path, outdir_region, rerun=False)
            if surface_fixed == False:
                surface_fixed = fix_surface(subject, region, -50, surface_path, outdir_region, rerun=True)
                if surface_fixed == False:
                    with open(os.path.join(outdir_region, "errors.txt"), "a") as f:
                        now = datetime.now().strftime('%d-%m-%Y %H:%M')
                        f.write(f"[ Error | {now} ] Surface fixing for {region} failed after two attempts {subject}\n")
        
        # -------------------- GLOBAL MESH --------------------
        surface_path = os.path.join(SURFACE_DIR,  subject, "outputs", "surfaces", "global.stl")
        outdir_global = os.path.join(subject_outdir, "global")
        os.makedirs(os.path.dirname(outdir_global), exist_ok=True)
        global_vtk_path = os.path.join(outdir_global, "global.vtk")
        results_path = os.path.join(outdir_global, "results.txt")
        start_coarse = REGIONS["global"]

        # Global mesh already exists
        if os.path.isfile(global_vtk_path):
            # Results.txt missing
            if not os.path.isfile(results_path):
                fixed_surface = os.path.join(outdir_global, "global.stl")
                if os.path.isfile(fixed_surface):
                    surface_path = fixed_surface

                global_bbox, num_triangles, global_elements = get_parameters(surface_path, global_vtk_path)
                target_elements = TARGET_GLOBAL_ELEMENTS
                pct_diff = 100 * (global_elements / target_elements)
                with open(results_path, "a") as f:
                    f.write(f"Subject {subject} Region Global\n")
                    f.write(f"BoundingBoxSize {global_bbox:.0f}\n")
                    f.write(f"SurfaceElements {num_triangles}\n")
                    f.write(f"VolumeElements {global_elements}\n")
                    f.write(f"TargetElements {target_elements:.0f}\n")
                    f.write(f"PercentDiff {pct_diff:.0f}%\n")
            # Result.txt present, pull values
            else:
                with open(results_path, "r") as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    if line.startswith("BoundingBoxSize"):
                        global_bbox = float(line.split()[1])
                    if line.startswith("VolumeElements"):
                        global_elements = float(line.split()[1])

        # Global mesh does not exist
        else:
            surface_fixed = fix_surface(subject, "global", start_coarse, surface_path, outdir_global, rerun=False)
            if surface_fixed == False:
                surface_fixed = fix_surface(subject, "global", start_coarse, surface_path, outdir_global, rerun=True)
                if surface_fixed == False:
                    with open(os.path.join(subject_outdir, "errors.txt"), "a") as f:
                        now = datetime.now().strftime('%d-%m-%Y %H:%M')
                        f.write(f"[ Error | {now} ] Global surface fixing failed after two attempts {subject}\n")

            # Try multiple coarseness values
            surface_path = os.path.join(outdir_global, "global.stl")
            smallest_discrepancy = float("inf")
            closest_coarseness = start_coarse
            found_within_tol = False
            no_global_mesh = False

            for coarseness in range(start_coarse, start_coarse + COARSENESS_STEPS):
                # First meshing attempt
                num_elements, bbox_volume, ok = generate_mesh(
                    subject, "global", coarseness, surface_path, outdir_global,
                    target_elements=TARGET_GLOBAL_ELEMENTS
                )

                # No valid mesh - move onto next coarseness
                if not ok:
                    continue
                else:
                    # Mesh produced assess element count vs target
                    discrepancy = abs(num_elements - TARGET_GLOBAL_ELEMENTS)
                    if discrepancy <= (TARGET_GLOBAL_ELEMENTS * TOLERANCE_FRAC):
                        # Accept and stop iterating
                        found_within_tol = True
                        global_elements = num_elements
                        global_bbox = float(bbox_volume)
                        with open(results_path, "a") as f:
                            now = datetime.now().strftime('%d-%m-%Y %H:%M')
                            f.write(f"\n[ Log | {now} ] SELECTED MESH COARSENESS: {coarseness} \n")
                        break
                    else:
                        # Not within tolerance - record closest
                        if discrepancy < smallest_discrepancy:
                            smallest_discrepancy = discrepancy
                            closest_coarseness = coarseness
                        # Move onto next coarseness
                        if os.path.exists(outdir_global):
                            for f in os.listdir(outdir_global):
                                if f.endswith((".vtk")):
                                    os.remove(os.path.join(outdir_global, f))
                        continue

            # If not found within tolerance, use the closest coarseness and generate once more
            if not found_within_tol:
                if not smallest_discrepancy == float("inf"):
                    num_elements, bbox_volume, ok = generate_mesh(
                        subject, "global", closest_coarseness, surface_path, outdir_global,
                        target_elements=TARGET_GLOBAL_ELEMENTS
                    )

                    with open(results_path, "a") as f:
                        now = datetime.now().strftime('%d-%m-%Y %H:%M')
                        f.write(f"\n[ Log | {now} ] SELECTED MESH COARSENESS: {closest_coarseness} \n")

                # Global mesh failed
                if not ok or smallest_discrepancy == float("inf"):
                    no_global_mesh = True
                    with open(os.path.join(subject_outdir, "errors.txt"), "a") as f:
                        now = datetime.now().strftime('%d-%m-%Y %H:%M')
                        f.write(f"[ Error | {now} ] Global meshing failed for subject {subject}\n")
                else:
                    global_elements = num_elements
                    global_bbox = float(bbox_volume)

            # If no global mesh, skip regions and move onto next subject
            if no_global_mesh:
                continue


        # -------------------- REGIONAL MESHES --------------------
        for region in REGIONS:
            if region == "global":
                continue
            start_coarse = REGIONS[region]
            surface_path = os.path.join(SURFACE_DIR, subject, "outputs", "surfaces", f"{region}.stl")
            outdir_region = os.path.join(subject_outdir, region)
            os.makedirs(os.path.dirname(outdir_region), exist_ok=True)
            region_vtk_path = os.path.join(outdir_region, f"{region}.vtk")
            results_path = os.path.join(outdir_region, "results.txt")

            # Regional mesh already exists
            if os.path.isfile(region_vtk_path):
                # Results.txt missing
                if not os.path.isfile(results_path):
                    fixed_surface = os.path.join(outdir_region, f"{region}.stl")
                    if os.path.isfile(fixed_surface):
                        surface_path = fixed_surface
                    bbox_volume, num_triangles, num_elements = get_parameters(surface_path, region_vtk_path)
                    target_elements = ((bbox_volume / global_bbox) * global_elements) * 0.5
                    pct_diff = 100 * (num_elements / target_elements)
                    with open(results_path, "a") as f:
                        f.write(f"Subject {subject} Region {region}\n")
                        f.write(f"BoundingBoxSize {bbox_volume:.0f}\n")
                        f.write(f"SurfaceElements {num_triangles}\n")
                        f.write(f"VolumeElements {num_elements}\n")
                        f.write(f"TargetElements {target_elements:.0f}\n")
                        f.write(f"PercentDiff {pct_diff:.0f}%\n")
                # Result.txt present, continue to next region
                else:
                    continue

            # Regional mesh does not exist
            else:
                surface_fixed = fix_surface(subject, region, start_coarse, surface_path, outdir_region, rerun=False)
                if surface_fixed == False:
                    surface_fixed = fix_surface(subject, region, start_coarse, surface_path, outdir_region, rerun=True)
                    if surface_fixed == False:
                        with open(os.path.join(outdir_region, "errors.txt"), "a") as f:
                            now = datetime.now().strftime('%d-%m-%Y %H:%M')
                            f.write(f"[ Error | {now} ] Surface fixing failed after two attempts {subject} {region}\n")

                surface_path = os.path.join(outdir_region, f"{region}.stl")

                bbox_volume, num_triangles = get_parameters(surface_path)
                target_elements = ((bbox_volume / global_bbox) * global_elements) * 0.5

                smallest_discrepancy = float("inf")
                closest_coarseness = start_coarse
                found_within_tol = False
                no_region_mesh = False

                for coarseness in range(start_coarse, start_coarse + COARSENESS_STEPS):
                    # Try to mesh
                    num_elements, bbox_volume, ok = generate_mesh(
                        subject, region, coarseness, surface_path, outdir_region,
                        target_elements=target_elements
                    )

                    # No valid mesh - move onto next coarseness
                    if not ok:
                        continue

                    # Mesh produced assess element count vs target
                    discrepancy = abs(num_elements - target_elements)
                    if discrepancy <= (target_elements * TOLERANCE_FRAC):
                        # Accept and stop iterating
                        found_within_tol = True
                        with open(results_path, "a") as f:
                            now = datetime.now().strftime('%d-%m-%Y %H:%M')
                            f.write(f"\n[ Log | {now} ] SELECTED MESH COARSENESS: {coarseness} \n")
                        break
                    else:
                        # Not within tolerance - record closest
                        if discrepancy < smallest_discrepancy:
                            smallest_discrepancy = discrepancy
                            closest_coarseness = coarseness
                        # Move onto next coarseness
                        if os.path.exists(outdir_region):
                            for f in os.listdir(outdir_region):
                                if f.endswith((".vtk")):
                                    os.remove(os.path.join(outdir_region, f))
                        continue

                # If not found within tolerance, use the closest coarseness and generate once more
                if not found_within_tol:
                    if not smallest_discrepancy == float("inf"):
                        num_elements, bbox_volume, ok = generate_mesh(
                            subject, region, closest_coarseness, surface_path, outdir_region,
                            target_elements=target_elements
                        )

                        with open(results_path, "a") as f:
                            now = datetime.now().strftime('%d-%m-%Y %H:%M')
                            f.write(f"\n[ Log | {now} ] SELECTED MESH COARSENESS: {closest_coarseness} \n")

                    # Region mesh failed
                    if not ok or smallest_discrepancy == float("inf"):
                        with open(os.path.join(subject_outdir, "errors.txt"), "a") as f:
                            now = datetime.now().strftime('%d-%m-%Y %H:%M')
                            f.write(f"[ Error | {now} ] Region meshing failed for subject {subject} {region}\n")
                        break

        # -------------------- FINAL SUBJECT CHECK --------------------
        subject_success = True
        for region in REGIONS:
            vtk_file = os.path.join(subject_outdir, region, f"{region}.vtk")
            if not os.path.isfile(vtk_file):
                subject_success = False
                # append error lines
                with open(os.path.join(subject_outdir, "errors.txt"), "a" if os.path.exists(os.path.join(subject_outdir, "errors.txt")) else "w") as ef:
                    now = datetime.now().strftime('%d-%m-%Y %H:%M')
                    ef.write(f"[ Error | {now} ] No mesh produced for {subject} {region}\n")

        if subject_success:
            with open(os.path.join(subject_outdir, "results.txt"), "a") as rf:
                now = datetime.now().strftime('%d-%m-%Y %H:%M')
                rf.write(f"[ Log | {now} ] All meshes produced successfully\n")

if __name__ == "__main__":
    main()