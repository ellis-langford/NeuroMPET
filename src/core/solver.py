# Imports
import os
import sys
import glob
import tempfile
import subprocess
import numpy as np
import pandas as pd
import pyvista as pv
import meshio
from collections import Counter
from scipy.spatial import KDTree

class Solver(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributes are present
        to_inherit = [
            "loggers",
            "parameters",
            "input_dir",
            "interim_dir",
            "output_dir",
            "log_dir",
            "surface_dir",
            "mesh_dir",
            "solver_labels_file",
            "bc_file",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)
    
    def resolve_unique_path(self, search_dir: str, filename: str) -> str:
        """
        Resolve an exact file match within a directory tree.

        Parameters:
        ---
        search_dir (str)   : Directory to search recursively.
        filename (str)     : Exact filename to resolve.

        Returns:
        ---
        str: Full path to the uniquely matched file.
        """
        matches = glob.glob(os.path.join(search_dir, "**", filename), recursive=True)
        if len(matches) != 1:
            self.loggers.errors(
                f"Expected exactly one {filename} in {search_dir}, found {matches}"
            )
        return matches[0]

    def read_mesh_with_fallback(self, vtk_path: str) -> meshio.Mesh:
        """
        Read a mesh file with a legacy VTK datatype fallback.

        Parameters:
        ---
        vtk_path (str)   : Path to the mesh file.

        Returns:
        ---
        meshio.Mesh: The read mesh object.
        """
        try:
            return meshio.read(vtk_path)
        except Exception as e:
            try:
                with open(vtk_path, "r") as f:
                    content = f.read()
                content = content.replace("unsigned_char", "int")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                try:
                    return meshio.read(temp_path)
                finally:
                    os.remove(temp_path)
            except Exception as fallback_error:
                self.loggers.errors(
                    f"Failed to read mesh file {vtk_path}: {e}. "
                    f"Fallback VTK read also failed: {fallback_error}"
                )

    def compute_stl_volume(self, surface: pv.PolyData) -> float:
        """
        Compute enclosed STL volume from triangulated surface faces.

        Parameters:
        ---
        surface (pv.PolyData)   : Triangulated closed surface.

        Returns:
        ---
        float: Enclosed volume in cubic millimeters.
        """
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        triangles = np.asarray(surface.points[faces], dtype=float)
        volume = 0.0
        for triangle in triangles:
            a, b, c = triangle
            volume += np.dot(a, np.cross(b, c))
        return abs(volume) / 6.0

    def scale_boundary_conditions(self):
        """
        Scale boundary conditions to the enclosed global surface volume.
        """
        # Read and validate global surface
        surface = pv.read(self.global_surface).triangulate().clean()
        if not surface.is_manifold or surface.n_open_edges != 0:
            self.loggers.errors(
                f"Global surface {self.global_surface} is not a closed manifold surface"
            )
        pyvista_volume = abs(float(surface.volume))
        triangle_volume = self.compute_stl_volume(surface)
        if pyvista_volume <= 0.0 or triangle_volume <= 0.0:
            self.loggers.errors(
                f"Computed non-positive enclosed volume from global surface {self.global_surface}"
            )

        # Check volumes are consistent within 1%
        relative_difference = abs(pyvista_volume - triangle_volume) / triangle_volume
        if relative_difference > 0.01:
            self.loggers.errors(
                "Global surface volume estimates disagree by more than 1%: "
                f"PyVista={pyvista_volume:.6f} mm^3, "
                f"triangle={triangle_volume:.6f} mm^3"
            )

        volume = triangle_volume / 1000.0

        # Read header line manually
        with open(self.bc_file, "r") as f:
            header_line = f.readline().strip()
    
        df = pd.read_csv(self.bc_file, skiprows=1, header=None)
    
        # Scale odd columns
        for col_index in range(1, df.shape[1], 2):
            df.iloc[:, col_index] *= volume
    
        # Write back original header exactly
        scaled_fpath = os.path.join(os.path.dirname(self.bc_file), "boundary_conditions_scaled.csv")
        with open(scaled_fpath, "w") as f:
            f.write(header_line + "\n")
            df.to_csv(f, index=False, header=False)

        self.bc_file = scaled_fpath

    def build_solver_command(self) -> str:
        """
        Build the MPET solver command.

        Returns:
        ---
        str: The complete command to run the MPET solver.
        """
        # Define parameters from config
        self.timestep_size = self.parameters["timestep_size"]
        self.waveform_timesteps = self.parameters["waveform_timesteps"]
        self.num_waveforms = self.parameters["num_waveforms"]
        self.output_timestep_interval = self.parameters["output_timestep_interval"]

        # Outdir
        self.modelling_outdir = os.path.join(self.output_dir, "modelling")
        os.makedirs(self.modelling_outdir, exist_ok=True)

        # Define command
        self.solver_command = f"make clean && make -s && " + \
                              f"./MPET3D '{self.bit_file}' '{self.modelling_outdir}' " + \
                              f"{self.timestep_size} {self.waveform_timesteps} " + \
                              f"{self.num_waveforms} {self.output_timestep_interval} " + \
                              f"'{self.bc_file}' '{self.solver_labels_file}'"

        # Define location of MPET source code
        self.source_code_dir = "/app/opt/mpet_source_code"
        return self.solver_command

    def convert_and_export_custom_mesh(self, vtk_path: str, wholebrain_stl: str,
                                       ventricles_stl: str, output_file: str) -> str:
        """
        Convert a tetrahedral mesh and STL surfaces into a custom .bit mesh file.

        Parameters:
        ---
        vtk_path (str)          : Path to the global mesh file.
        wholebrain_stl (str)    : Path to the wholebrain STL surface.
        ventricles_stl (str)    : Path to the ventricles STL surface.
        output_file (str)       : Path to save the custom .bit file.

        Returns:
        ---
        str: Path to the generated .bit file.
        """
        # Load mesh and extract points
        mesh = self.read_mesh_with_fallback(vtk_path)
        points = np.asarray(mesh.points, dtype=float)
        tets_idx = None
        for cell_block in mesh.cells:
            if cell_block.type == "tetra":
                tets_idx = np.asarray(cell_block.data, dtype=int)
                break
            if cell_block.type == "tetra10":
                tets_idx = np.asarray(cell_block.data[:, :4], dtype=int)
                break

        if tets_idx is None:
            self.loggers.errors(
                f"No supported cell block ['tetra', 'tetra10'] found in mesh file {vtk_path}"
            )

        # Load STL surfaces
        wb_surf = pv.read(wholebrain_stl)
        vent_surf = pv.read(ventricles_stl)

        # Identify faces on surface of tetrahedral mesh
        faces = []
        for tet in tets_idx:
            n0, n1, n2, n3 = tet
            faces.extend([
                tuple(sorted([n0, n1, n2])),
                tuple(sorted([n0, n1, n3])),
                tuple(sorted([n0, n2, n3])),
                tuple(sorted([n1, n2, n3])),
            ])
        face_count = Counter(faces)
        surface_faces = np.array([f for f, count in face_count.items() if count == 1], dtype=int)

        # Build KD-Trees for STL surfaces
        tree_outer = KDTree(wb_surf.points)
        tree_inner = KDTree(vent_surf.points)
        face_centroids = points[surface_faces].mean(axis=1)

        # Compute distances to inner and outer surfaces
        dist_outer, _ = tree_outer.query(face_centroids)
        dist_inner, _ = tree_inner.query(face_centroids)

        # Classify faces as inner or outer based on distance
        threshold_mm = 0.5
        dist_diff = dist_outer - dist_inner
        prefer_inner = np.abs(dist_diff) < threshold_mm
        initial_inner = dist_inner < dist_outer
        inner_mask = initial_inner | prefer_inner
        outer_mask = ~inner_mask

        inner_faces = surface_faces[inner_mask]
        outer_faces = surface_faces[outer_mask]

        # Write to custom .bit file
        with open(output_file, "w") as f:
            f.write("$Node\n")
            f.write(f"{len(points)}\n")
            for point_id, (x, y, z) in enumerate(points, start=1):
                f.write(f"{point_id} {x:.6e} {y:.6e} {z:.6e}\n")

            f.write("$OuterFaceCell\n")
            f.write(f"{len(outer_faces)}\n")
            for tri in outer_faces:
                f.write(f"3 o {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

            f.write("$InnerFaceCell\n")
            f.write(f"{len(inner_faces)}\n")
            for tri in inner_faces:
                f.write(f"3 i {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

            f.write("$TetraCell\n")
            f.write(f"{len(tets_idx)}\n")
            for tet in tets_idx:
                n0, n1, n2, n3 = tet + 1
                f.write(f"4 {n0} {n1} {n2} {n3}\n")

        return output_file

    def extract_surfaces_from_bit(self, bit_file: str, outdir: str) -> tuple[str, str]:
        """
        Extract inner and outer surfaces from a .bit file for visualisation.

        Parameters:
        ---
        bit_file (str)   : Path to the .bit mesh file.
        outdir (str)     : Directory to save the inner and outer surface files.

        Returns:
        ---
        tuple[str, str]: Paths to the generated inner and outer surface VTU files.
        """
        nodes = []
        inner_faces = []
        outer_faces = []

        # Read .bit file and extract nodes and faces
        with open(bit_file, "r") as f:
            lines = f.readlines()

        line_index = 0
        while line_index < len(lines):
            line = lines[line_index].strip()

            if line == "$Node":
                count = int(lines[line_index + 1])
                for offset in range(count):
                    parts = lines[line_index + 2 + offset].split()
                    nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
                line_index += 2 + count
                continue

            if line == "$OuterFaceCell":
                count = int(lines[line_index + 1])
                for offset in range(count):
                    parts = lines[line_index + 2 + offset].split()
                    n1, n2, n3 = map(int, parts[2:5])
                    outer_faces.append([n1 - 1, n2 - 1, n3 - 1])
                line_index += 2 + count
                continue

            if line == "$InnerFaceCell":
                count = int(lines[line_index + 1])
                for offset in range(count):
                    parts = lines[line_index + 2 + offset].split()
                    n1, n2, n3 = map(int, parts[2:5])
                    inner_faces.append([n1 - 1, n2 - 1, n3 - 1])
                line_index += 2 + count
                continue

            line_index += 1

        # Convert to numpy arrays
        points = np.asarray(nodes, dtype=float)
        inner_tri = np.asarray(inner_faces, dtype=np.int64).reshape(-1, 3)
        outer_tri = np.asarray(outer_faces, dtype=np.int64).reshape(-1, 3)

        if inner_tri.shape[0] == 0:
            self.loggers.plugin_log("No inner surface triangles found in .bit file")
        if outer_tri.shape[0] == 0:
            self.loggers.plugin_log("No outer surface triangles found in .bit file")

        # Write inner and outer surfaces to VTU files for visualisation
        inner_surface_file = os.path.join(outdir, "inner_surface.vtu")
        outer_surface_file = os.path.join(outdir, "outer_surface.vtu")
        meshio.write(
            inner_surface_file,
            meshio.Mesh(points=points, cells=[("triangle", inner_tri)]),
        )
        meshio.write(
            outer_surface_file,
            meshio.Mesh(points=points, cells=[("triangle", outer_tri)]),
        )
        return inner_surface_file, outer_surface_file

    def run_solver(self):
        """
        Run the MPET solver.
        """
        self.loggers.plugin_log("Starting execution")
        self.solver_log = os.path.join(self.log_dir, "solver.log")
        self.loggers.plugin_log(f"Solver command: {self.solver_command}")
        with open(self.solver_log, "w") as outfile:
            solver_sub = subprocess.run(["bash", "-c",
                                            self.solver_command],
                                            cwd=self.source_code_dir,
                                            stdout=outfile,
                                            stderr=subprocess.STDOUT)
            
        if solver_sub.returncode != 0:
            self.loggers.errors(f"Solver execution returned non-zero exit status - " +
                                f"please check log file at {self.solver_log}")

        # Check required outputs have been produced
        for timestep in range(0, ((self.waveform_timesteps * self.num_waveforms) + self.output_timestep_interval), self.output_timestep_interval):
            result_file = os.path.join(self.modelling_outdir, f"outputs_{timestep}.vtu")

            if not os.path.exists(result_file):
                self.loggers.errors(f"Solver has not produced an output file at timestep {timestep} " +
                                    f"- please check log file at {self.solver_log}")
        # Check regional file        
        if not os.path.exists(os.path.join(self.modelling_outdir, f"outputs_region.vtu")):
            self.loggers.errors(f"Solver has not produced a regional output file " +
                                f"- please check log file at {self.solver_log}")

    def run_modelling(self):
        """
        Run modelling processing.
        """
        self.interim_dir = os.path.join(self.interim_dir, "modelling")
        os.makedirs(self.interim_dir, exist_ok=True)

        # Define mesh and surface files
        self.mesh_file = self.resolve_unique_path(self.mesh_dir, "global.vtk")
        self.global_surface = self.resolve_unique_path(self.surface_dir, "global.stl")
        self.wb_surface = self.resolve_unique_path(self.surface_dir, "wholebrain.stl")
        self.vent_surface = self.resolve_unique_path(self.surface_dir, "ventricles.stl")

        # Scale boundary conditions to brain size
        self.loggers.plugin_log("Scaling boundary condition file")
        self.scale_boundary_conditions()

        # Produce custom .bit file
        self.loggers.plugin_log("Creating custom .bit file")
        self.bit_file = os.path.join(self.input_dir, "global.bit")
        self.convert_and_export_custom_mesh(self.mesh_file,
                                            self.wb_surface,
                                            self.vent_surface,
                                            self.bit_file)
        if not os.path.isfile(self.bit_file):
            self.loggers.errors("Failed to generate custom .bit file")
        
        # Produce inner and outer vtu files for visualisation
        self.extract_surfaces_from_bit(self.bit_file,
                                       self.interim_dir)

        # Run solver
        self.loggers.plugin_log("Running MPET solver")
        self.build_solver_command()
        self.run_solver()
