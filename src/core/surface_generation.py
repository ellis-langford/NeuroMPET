# Imports
import os
import sys
import numpy as np
import nibabel as nib
import pyvista as pv
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_fill_holes, generate_binary_structure
from scipy.spatial import KDTree
from collections import deque

class SurfaceGen(object):
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
            "segmentation_dir",
            "regions",
            "region_definitions",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def reverse_face_winding(self, surface: pv.PolyData) -> pv.PolyData:
        """
        Reverse polygon winding for every face in a PolyData surface.

        Parameters:
        ---
        surface (pv.PolyData) : Input surface to reverse face winding.

        Returns:
        ---
        pv.PolyData : Surface with reversed face winding.
        """
        # Create array of reversed faces
        flipped = surface.copy()
        faces = np.asarray(flipped.faces, dtype=np.int64)

        # If no faces, return original surface
        if faces.size == 0:
            return flipped

        # Iterate through faces and reverse the order of vertices for each face
        reversed_faces = []
        i = 0
        while i < faces.size:
            npts = int(faces[i])
            cell = faces[i + 1 : i + 1 + npts]
            reversed_faces.extend([npts, *cell[::-1]])
            i += npts + 1

        # Update the faces of the flipped surface with the reversed faces
        flipped.faces = np.asarray(reversed_faces, dtype=faces.dtype)
        return flipped

    def orient_surface(self, surface: pv.PolyData, prefer_inward: bool = False) -> pv.PolyData:
        """
        Enforce consistent face orientation before export.

        For standalone surfaces the expected orientation is outward. For inner
        cavity shells in combined composite exports the expected orientation is inward.

        Parameters:
        ---
        surface (pv.PolyData) : Input surface to orient.
        prefer_inward (bool)  : Whether to prefer inward-facing or outward-facing normals.

        Returns:
        ---
        pv.PolyData : Surface with enforced consistent orientation.
        """
        oriented = surface.triangulate().clean()

        # Compute normals
        oriented = oriented.compute_normals(
            cell_normals=True,
            point_normals=True,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=False,
        )

        # If no faces/cells, return original surface
        if oriented.n_cells == 0:
            return oriented

        # Determine overall orientation of surface
        centres = oriented.cell_centers().points  # Get cell centres
        normals = np.asarray(oriented.cell_normals, dtype=float)  # Get cell normals
        centre = np.asarray(oriented.center, dtype=float)  # Get overall surface centre
        radial = np.asarray(centres, dtype=float) - centre  # Compute radial vectors from centre to cell centres
        valid = np.linalg.norm(radial, axis=1) > 1e-12  # Avoid division by zero

        # Compute mean dot product between normals and radial vectors to determine if surface is overall inward or outward facing
        if np.any(valid):
            # Positive score, consistent normal with radial vectors (outward), negative (inward)
            orientation_score = float(np.mean(np.einsum("ij,ij->i", normals[valid], radial[valid])))
            # Determine if we need to flip the surface
            should_flip = orientation_score > 0.0 if prefer_inward else orientation_score < 0.0

            # Flip the surface
            if should_flip:
                oriented = self.reverse_face_winding(oriented)
                oriented = oriented.compute_normals(
                    cell_normals=True,
                    point_normals=True,
                    consistent_normals=True,
                    auto_orient_normals=False,
                    inplace=False,
                )

        return oriented.clean()

    def segmentation_to_surface(self, segmentation: np.ndarray, affine: np.ndarray, smooth_sigma: float = 0.7) -> pv.PolyData:
        """
        Convert a binary segmentation to a PyVista surface.

        Parameters:
        ---
        segmentation (array)      : Binary segmentation data.
        affine (array)            : Affine matrix for the segmentation.
        smooth_sigma (float)      : Smoothing value to apply.

        Returns:
        ---
        pv.PolyData : Generated surface from the segmentation.
        """
        # Apply Gaussian smoothing to reduce small holes/gaps before marching cubes
        data = gaussian_filter(segmentation.astype(float), sigma=smooth_sigma)

        # Check range to avoid division by zero during normalisation
        data_range = data.max() - data.min()
        if data_range == 0:
            self.loggers.errors("Surface generation failed because the segmentation has zero intensity range")

        # Normalise data to 0-1 range for marching cubes
        data = (data - data.min()) / data_range

        # Extract surface using marching cubes
        verts, faces, _, _ = measure.marching_cubes(data, level=0.5)

        # Transform vertices to world coordinates using affine
        verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1))])
        verts_world = (affine @ verts_hom.T).T[:, :3]

        # Convert faces to PyVista format
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        surface = pv.PolyData(verts_world, faces_pv)
        surface = surface.connectivity(extraction_mode='largest')  # Remove disconnected pieces
        surface = surface.clean()  # Remove duplicates and degenerate faces
        
        return surface

    def resolve_segmentation_path(self, region: str) -> str:
        """
        Resolve the exact staged segmentation file path for a region.

        Parameters:
        ---
        region (str)   : Region name to resolve.

        Returns:
        ---
        str : Resolved file path for the region's segmentation.
        """
        seg_path = os.path.join(self.segmentation_dir, f"{region}_bin.nii.gz")
        fallback_seg_path = os.path.join(self.segmentation_dir, f"{region}.nii.gz")

        if os.path.isfile(seg_path):
            return seg_path
        if os.path.isfile(fallback_seg_path):
            return fallback_seg_path

        self.loggers.errors(
            f"A segmentation for {region} must be provided at {seg_path} or {fallback_seg_path}"
        )

    def generate_combined_surface(self, outer_region: str, inner_region: str,
                                  output_prefix: str, save_component_surfaces: bool = True):
        """
        Generate inner and outer component surfaces and a combined composite surface.

        Example:
        - global: outer_region=wholebrain, inner_region=ventricles, output_prefix=global
        - cerebrumGM_L: outer_region=cerebrum_L, inner_region=cerebrumWM_L, output_prefix=cerebrumGM_L

        Parameters:
        ---
        outer_region (str)                : Region to use as outer surface.
        inner_region (str)                : Region to use as inner surface to subtract from outer.
        output_prefix (str)               : Prefix for output surface file.
        save_component_surfaces (bool)    : Whether to save the individual component surfaces.
        """
        self.loggers.verbose_log(f"Generating combined surface {output_prefix} from "
                                 f"outer={outer_region} and inner={inner_region}")

        # Load outer region
        outer_bin = self.resolve_segmentation_path(outer_region)
        outer_nii = nib.load(outer_bin)
        outer_data = outer_nii.get_fdata().astype(bool)
        outer_affine = outer_nii.affine

        # Load inner region
        inner_bin = self.resolve_segmentation_path(inner_region)
        inner_nii = nib.load(inner_bin)
        inner_data = inner_nii.get_fdata().astype(bool)
        
        # Remove inner region oversegmentation
        inner_data = inner_data & outer_data
        
        # Fill holes in outer region
        visited = np.zeros_like(outer_data, dtype=bool)  # Create a mask of visited voxels
        # Initialise queue with the 8 corners of the volume
        q = deque()
        shape = outer_data.shape
        corners = [(0,0,0), (0,0,shape[2]-1), (0,shape[1]-1,0), (0,shape[1]-1,shape[2]-1),
                   (shape[0]-1,0,0), (shape[0]-1,0,shape[2]-1), (shape[0]-1,shape[1]-1,0), (shape[0]-1,shape[1]-1,shape[2]-1)]
        for c in corners:
            if not outer_data[c]:
                q.append(c)
                visited[c] = True
        
        # 6-connectivity flood-fill
        neighbours = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in neighbours:
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    if not outer_data[nx,ny,nz] and not visited[nx,ny,nz]:
                        visited[nx,ny,nz] = True
                        q.append((nx,ny,nz))
        
        # All voxels not reachable from outside are interior holes
        interior_holes = (~visited) & (~outer_data)
        outer_data[interior_holes] = True  # Fill them

        # Close diagonal pinholes & single-voxel gaps
        struct26 = generate_binary_structure(3, 2)
        outer_data = binary_closing(outer_data, structure=struct26, iterations=1)

        # Separate touching surfaces
        radius = 2
        max_iter = 5
        g = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]  # Spherical structuring element
        struct = (g[0]**2 + g[1]**2 + g[2]**2) <= radius**2

        # Try several times to separate
        for i in range(max_iter):
            # Detect touching voxels
            touching = inner_data & outer_data
            if not np.any(touching):
                break
            # Dilate the touching region
            localised_growth = binary_dilation(touching, structure=struct)
            outer_data |= localised_growth  # Grow wholebrain mask locally

        # Pad volumes to prevent open boundaries in marching cubes
        pad = 1
        outer_data = np.pad(outer_data, pad, mode="constant", constant_values=False)
        inner_data = np.pad(inner_data, pad, mode="constant", constant_values=False)
    
        outer_affine = outer_affine.copy()
        outer_affine[:3, 3] -= pad * np.diag(outer_affine)[:3]

        # Generate surfaces
        outer_surf = self.segmentation_to_surface(outer_data, outer_affine)
        inner_surf = self.segmentation_to_surface(inner_data, outer_affine)
        
        # Check for overlaps and fix at surface level
        tolerance = 0.1
        max_iter = 5
        tree = KDTree(outer_surf.points)
        dists, idx = tree.query(inner_surf.points)
        
        if np.any(dists < tolerance):    
            # Ensure normals exist
            if outer_surf.point_normals is None:
                outer_surf.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    
            iteration = 0
            while iteration < max_iter:
                # Find closest cortical points to ventricle points
                inner_points = inner_surf.points
                tree = KDTree(outer_surf.points)
                distances, idx = tree.query(inner_points)
                distances = np.array(distances)
                idx = np.array(idx)
                
                # Identify points that are too close
                touching_mask = distances < tolerance
                if not np.any(touching_mask):
                    # No overlaps left
                    break
                
                # Compute minimal displacement
                displacement = (tolerance - distances[touching_mask])[:, np.newaxis]
                normals = outer_surf.point_normals[idx[touching_mask]]
                outer_surf.points[idx[touching_mask]] += normals * displacement
                
                iteration += 1
            
            # Check again for overlaps
            tree = KDTree(outer_surf.points)
            distances, _ = tree.query(inner_surf.points)
            if np.any(distances < 0.1):
                self.loggers.errors(f"Composite surface generation failed for {output_prefix} - overlaps present")

        # Final watertightness surface repair
        if not outer_surf.is_manifold:
            outer_surf = outer_surf.clean(
                tolerance=1e-6,
                inplace=False,
            )

        # Check watertight
        if not outer_surf.is_all_triangles:
            outer_surf.triangulate(inplace=True)
        if not outer_surf.is_manifold:
            self.loggers.errors(f"Outer surface generation failed for {outer_region} - not watertight")
        if not inner_surf.is_all_triangles:
            inner_surf.triangulate(inplace=True)
        if not inner_surf.is_manifold:
            self.loggers.errors(f"Inner surface generation failed for {inner_region} - not watertight")

        # Enforce consistent orientation, outer outward-facing, inner cavity inward-facing
        outer_surf = self.orient_surface(outer_surf, prefer_inward=False)
        inner_surf = self.orient_surface(inner_surf, prefer_inward=False)
        inner_composite_surf = self.orient_surface(inner_surf.copy(), prefer_inward=True)

        # Save outer/inner surfaces
        if save_component_surfaces:
            outer_surf.save(os.path.join(self.output_dir, "surfaces", f"{outer_region}.stl"))
            inner_surf.save(os.path.join(self.output_dir, "surfaces", f"{inner_region}.stl"))

        # Build combined surface
        composite_surf = outer_surf.merge(inner_composite_surf, merge_points=False)

        # Save combined shell
        outpath = os.path.join(self.output_dir, "surfaces", f"{output_prefix}.stl")
        composite_surf.save(outpath, binary=False)

        # Final check for combined surface
        if not os.path.exists(outpath):
            self.loggers.errors(f"Composite shell generation failed for {output_prefix}")

    def generate_region_surface(self, region: str):
        """
        Generate a standalone regional surface file.

        Parameters:
        ---
        region (str)   : Region to generate a surface for.
        """
        self.loggers.verbose_log(f"Generating standalone surface for {region}")

        # Load data
        bin_data = self.resolve_segmentation_path(region)
        region_nii = nib.load(bin_data)
        data = region_nii.get_fdata()
        affine = region_nii.affine

        # Pre-fix binary mask to reduce tiny holes/gaps before marching cubes
        mask = data.astype(bool)
        struct26 = generate_binary_structure(3, 2)
        mask = binary_closing(mask, structure=struct26, iterations=1)
        mask = binary_fill_holes(mask)

        # Generate surface and enforce consistent orientation
        surface = self.segmentation_to_surface(mask, affine, smooth_sigma=0.7)
        surface = self.orient_surface(surface, prefer_inward=False)

        # Enforce triangle-only geometry before manifold/watertight checks
        if not surface.is_all_triangles:
            surface = surface.triangulate()

        # Iterative repair for non-manifold/open-edge surfaces
        for hole_size in [5.0, 20.0, 50.0, 100.0]:
            if surface.is_manifold and surface.n_open_edges == 0:
                break
            try:
                surface = surface.fill_holes(hole_size=hole_size, inplace=False)
            except Exception as e:
                self.loggers.verbose_log(f"Hole-filling attempt failed for {region} at size {hole_size}: {e}")
            surface = surface.clean(tolerance=1e-6, inplace=False)
            if not surface.is_all_triangles:
                surface = surface.triangulate()

        # Check for watertightness and manifoldness
        if not surface.is_manifold or surface.n_open_edges > 0:
            self.loggers.errors(
                f"Surface generation failed for {region} - not watertight/manifold "
                f"(is_manifold={surface.is_manifold}, open_edges={surface.n_open_edges})"
            )

        # Save surface
        outpath = os.path.join(self.output_dir, "surfaces", f"{region}.stl")
        surface.save(outpath, binary=False)

        # Final check for surface file
        if not os.path.exists(outpath):
            self.loggers.errors(f"Surface generation failed for {region}")

    def run_surface_gen(self):
        """
        Run surface STL generation.
        """
        self.loggers.plugin_log("Generating surface files")

        # Directories            
        self.interim_dir = os.path.join(self.interim_dir, "surface_generation")
        os.makedirs(self.interim_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "surfaces"), exist_ok=True)

        # Regions
        requested_regions = self.regions
        generated_regions = set()
        if "global" not in requested_regions:
            requested_regions = ["global", *requested_regions]

        # Generate surfaces from region definitions
        for region in requested_regions:
            if region in generated_regions:
                continue

            # Check region definition exists
            region_definition = self.region_definitions.get(region)
            if region_definition is None:
                self.loggers.errors(f"Surface generation requested for unknown region {region}")

            # Check if standalone segmentation surface generation is required
            if region_definition["region_type"] == "segmentation":
                self.generate_region_surface(region)
                generated_regions.add(region)
                continue

            # Lateral brainstem surfaces from splitting
            if region in ["brainstem_L", "brainstem_R"]:
                self.generate_region_surface(region)
                generated_regions.add(region)
                continue

            # Combined composite surface generation
            combine_regions = region_definition.get("combine_regions", [])
            subtract_regions = region_definition.get("subtract_regions", [])
            if len(combine_regions) != 1 or len(subtract_regions) != 1:
                self.loggers.errors(
                    f"Derived surface region {region} must define exactly one combine region and one subtract region"
                )

            outer_region = combine_regions[0]
            inner_region = subtract_regions[0]
            save_component_surfaces = region == "global"

            self.loggers.verbose_log(f"Generating derived combined surface for {region}")
            self.generate_combined_surface(
                outer_region=outer_region,
                inner_region=inner_region,
                output_prefix=region,
                save_component_surfaces=save_component_surfaces,
            )

            # Log generated regions
            generated_regions.add(region)
            if save_component_surfaces:
                generated_regions.update({outer_region, inner_region})
