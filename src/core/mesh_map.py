# Imports
import os
import sys
import glob
import tempfile
import nibabel as nib
import numpy as np
import pyvista as pv
import meshio
from scipy.spatial import KDTree


class MeshMap(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributes are present
        to_inherit = [
            "loggers",
            "parameters",
            "mesh_dir",
            "surface_dir",
            "interim_dir",
            "output_dir",
            "log_dir",
            "regions",
            "region_definitions",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def prepare_mesh_info_inputs(self):
        """
        Load the global mesh and region surfaces required for mesh mapping.
        """
        # Create mapping from region labels to solver labels for final label assignment
        self.region_label_to_solver_label = {}
        for region_name, region_definition in self.region_definitions.items():
            region_label = region_definition.get("region_label")
            solver_label = region_definition.get("solver_label")
            if region_label is not None and solver_label is not None:
                self.region_label_to_solver_label[int(region_label)] = int(solver_label)

        # Load global mesh
        global_matches = [
            fpath
            for fpath in glob.glob(os.path.join(self.mesh_dir, "**", "global.vtk"), recursive=True)
        ]
        if len(global_matches) == 0:
            self.loggers.errors(f"File global.vtk does not exist in {self.mesh_dir}")
        if len(global_matches) > 1:
            self.loggers.errors(
                f"Multiple files matched global.vtk in {self.mesh_dir}: {global_matches}"
            )
        self.global_mesh = global_matches[0]

        # Load global mesh with meshio, with fallback for unsigned char cell data if needed
        try:
            self.global_mesh_data = meshio.read(self.global_mesh)
        except Exception as e:
            try:
                with open(self.global_mesh, "r") as f:
                    content = f.read()
                content = content.replace("unsigned_char", "int")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".vtk", delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                try:
                    self.global_mesh_data = meshio.read(temp_path)
                finally:
                    os.remove(temp_path)
            except Exception as fallback_error:
                self.loggers.errors(
                    f"Failed to read mesh file {self.global_mesh}: {e}. "
                    f"Fallback VTK read also failed: {fallback_error}"
                )

        # Extract tetrahedral cell block and node coordinates
        tetra_cells = None
        for cell_block in self.global_mesh_data.cells:
            if cell_block.type == "tetra":
                tetra_cells = np.asarray(cell_block.data, dtype=int)
                break
            if cell_block.type == "tetra10":
                tetra_cells = np.asarray(cell_block.data[:, :4], dtype=int)
                break

        if tetra_cells is None:
            self.loggers.errors(
                f"No supported cell block ['tetra', 'tetra10'] found in mesh file {self.global_mesh}"
            )

        # Remap to only used points for efficiency
        used_points, inverse = np.unique(tetra_cells.ravel(), return_inverse=True)
        self.global_mesh_node_coords = np.asarray(self.global_mesh_data.points[used_points], dtype=float)
        self.global_mesh_cell_indices = inverse.reshape(tetra_cells.shape)
        self.global_mesh_tetra_indices = self.global_mesh_cell_indices

        # Load region surfaces and prepare mapping candidates, with hierarchy priority for resolving overlaps
        wholebrain_matches = [
            fpath
            for fpath in glob.glob(os.path.join(self.surface_dir, "**", "wholebrain.stl"), recursive=True)
        ]
        if len(wholebrain_matches) == 0:
            self.loggers.errors(f"File wholebrain.stl does not exist in {self.surface_dir}")
        if len(wholebrain_matches) > 1:
            self.loggers.errors(
                f"Multiple files matched wholebrain.stl in {self.surface_dir}: {wholebrain_matches}"
            )
        self.wholebrain_surface = wholebrain_matches[0]

        # Regions not used in mesh mapping
        excluded_regions = {
            "global",
            "wholebrain",
            "ventricles",
            "cerebrum_L",
            "cerebrum_R",
            "cerebellum_L",
            "cerebellum_R",
            "brainstem",
        }

        # Compute global mesh tetrahedra centroids for later classification
        self.mapping_candidates = []
        self.global_mesh_centroids = self.global_mesh_node_coords[self.global_mesh_tetra_indices, :3].mean(axis=1)

        for region in self.regions:
            if region in excluded_regions:
                continue

            # Check region definitions for required labels
            solver_label = self.region_definitions.get(region, {}).get("solver_label")
            region_label = self.region_definitions.get(region, {}).get("region_label")
            if solver_label is None:
                self.loggers.errors(f"No solver label defined for mesh-mapped region {region}")
            if region_label is None:
                self.loggers.errors(f"No region label defined for mesh-mapped region {region}")

            # Lower hierarchy_priority wins overlap conflicts.
            is_specific_gm = int(solver_label) in (1, 2) and region not in {"cerebrumGM_L", "cerebrumGM_R"}
            hierarchy_priority = 0 if is_specific_gm else 1

            # Find surface file for region
            surface_matches = [
                fpath
                for fpath in glob.glob(os.path.join(self.surface_dir, "**", f"{region}.stl"), recursive=True)
            ]
            if len(surface_matches) == 0:
                self.loggers.errors(f"File {region}.stl does not exist in {self.surface_dir}")
            if len(surface_matches) > 1:
                self.loggers.errors(
                    f"Multiple files matched {region}.stl in {self.surface_dir}: {surface_matches}"
                )
            self.mapping_candidates.append(
                {
                    "region": region,
                    "region_label": int(region_label),
                    "solver_label": int(solver_label),
                    "hierarchy_priority": hierarchy_priority,
                    "surface_path": surface_matches[0],
                }
            )

    def classify_tetrahedra(self):
        """
        Classify global mesh tetrahedra from region surfaces and hierarchy rules.
        """
        if not self.mapping_candidates:
            self.loggers.errors("No mesh-mapped regions available for label assignment")

        region_models = []
        for candidate in self.mapping_candidates:
            # Load and prepare surface for region
            region = candidate["region"]
            try:
                region_surface = pv.read(candidate["surface_path"]).triangulate().clean()
            except Exception as e:
                self.loggers.errors(f"Failed to read surface file {candidate['surface_path']}: {e}")

            # Compute surface scale for distance scoring, with fallback to small value if bounds are degenerate
            surface_points = np.asarray(region_surface.points, dtype=float)
            bounds = np.asarray(region_surface.bounds, dtype=float)
            surface_scale = float(
                np.linalg.norm([
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                ])
            )
            if surface_scale <= 0.0:
                surface_scale = 1e-12

            # Store region model information for later classification
            region_models.append(
                {
                    "region": region,
                    "region_label": candidate["region_label"],
                    "solver_label": candidate["solver_label"],
                    "hierarchy_priority": candidate["hierarchy_priority"],
                    "surface": region_surface,
                    "tree": KDTree(surface_points),
                    "scale": surface_scale,
                }
            )

        # Process in chunks to manage memory
        n_cells = len(self.global_mesh_centroids)
        region_cell_labels = np.zeros(n_cells, dtype=np.int32)
        chunk_size = 500_000
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = self.global_mesh_centroids[start:end]
            chunk_labels = np.zeros(len(chunk), dtype=np.int32)
            chunk_priority = np.full(len(chunk), np.inf, dtype=float)
            chunk_score = np.full(len(chunk), -np.inf, dtype=float)

            # Classify each tetrahedron in chunk against each region surface, applying hierarchy rules for overlaps
            for region_model in region_models:
                try:
                    enclosed = pv.PolyData(chunk).select_enclosed_points(
                        region_model["surface"],
                        tolerance=0.0,
                        check_surface=False,
                    )
                except Exception as e:
                    self.loggers.errors(
                        f"Failed to classify tetrahedra against surface {region_model['region']}: {e}"
                    )

                # Determine which tetrahedra are inside the region surface
                in_region = np.asarray(enclosed["SelectedPoints"]).astype(bool)
                if not np.any(in_region):
                    continue

                # For tetrahedra inside the region, compute distance scores to surface and apply hierarchy rules
                distances, _ = region_model["tree"].query(chunk, k=1)
                scores = distances / region_model["scale"]
                better_match = in_region & (
                    (region_model["hierarchy_priority"] < chunk_priority)
                    | (
                        (region_model["hierarchy_priority"] == chunk_priority)
                        & (scores > chunk_score)
                    )
                )

                # Update labels, priority, and score for better matches
                chunk_labels[better_match] = region_model["region_label"]
                chunk_priority[better_match] = region_model["hierarchy_priority"]
                chunk_score[better_match] = scores[better_match]

            region_cell_labels[start:end] = chunk_labels

        # Fill any unlabelled tetrahedra from the nearest labelled tetrahedron
        unlabelled_idx = np.where(region_cell_labels == 0)[0]
        labelled_idx = np.where(region_cell_labels != 0)[0]
        if labelled_idx.size == 0:
            self.loggers.errors("No labelled global mesh tetrahedra were produced during mesh mapping")

        if unlabelled_idx.size > 0:
            tree = KDTree(self.global_mesh_centroids[labelled_idx])
            _, nearest = tree.query(self.global_mesh_centroids[unlabelled_idx], k=1)
            region_cell_labels[unlabelled_idx] = region_cell_labels[labelled_idx[nearest]]

        self.region_cell_labels = region_cell_labels

    def finalise_cell_labels(self):
        """
        Finalise cell labels for detailed and solver outputs.
        """
        # Define cell labels, tetrahedral cell indices, and count total cells for logging
        cell_labels = self.region_cell_labels.copy()
        tets = self.global_mesh_cell_indices
        n_cells = len(tets)

        # Relabel WM tetrahedra on the outer wholebrain surface to GM
        wholebrain_surface = pv.read(self.wholebrain_surface)
        wholebrain_points = np.asarray(wholebrain_surface.points, dtype=float)
        tree_surface = KDTree(wholebrain_points)
        tree_nodes = KDTree(self.global_mesh_node_coords)

        # Determine surface tolerance
        surface_to_mesh_distances, _ = tree_nodes.query(wholebrain_points, k=1)
        surface_tolerance = float(np.percentile(surface_to_mesh_distances, 99))

        # Compute distance from surface whether closer than tolerance
        node_distances, _ = tree_surface.query(self.global_mesh_node_coords, k=1)
        outer_node_mask = node_distances <= surface_tolerance
        outer_cell_mask = np.any(outer_node_mask[tets], axis=1)

        # Relabel outer-surface WM tetrahedra to GM equivalent region
        relabel_map = {}
        gm_left = self.region_definitions["cerebrumGM_L"]["region_label"]
        gm_right = self.region_definitions["cerebrumGM_R"]["region_label"]
        wm_left = self.region_definitions["cerebrumWM_L"]["region_label"]
        wm_right = self.region_definitions["cerebrumWM_R"]["region_label"]
        relabel_map[wm_left] = gm_left
        relabel_map[wm_right] = gm_right
        gm_left = self.region_definitions["cerebellumGM_L"]["region_label"]
        gm_right = self.region_definitions["cerebellumGM_R"]["region_label"]
        wm_left = self.region_definitions["cerebellumWM_L"]["region_label"]
        wm_right = self.region_definitions["cerebellumWM_R"]["region_label"]
        relabel_map[wm_left] = gm_left
        relabel_map[wm_right] = gm_right

        # Count and apply relabeling
        relabelled_count = 0
        for wm_label, gm_label in relabel_map.items():
            relabel_mask = outer_cell_mask & (cell_labels == wm_label)
            relabelled_count += int(np.sum(relabel_mask))
            cell_labels[relabel_mask] = gm_label

        if relabelled_count > 0:
            self.loggers.verbose_log(
                f"Relabelled {relabelled_count} outer-surface WM tetrahedra to GM"
            )

        # Check for any zero-labelled tetrahedra
        self.region_cell_labels = cell_labels
        if np.any(self.region_cell_labels == 0):
            self.loggers.errors("Mesh mapping produced unlabelled tetrahedra with region label 0")

        # Map region labels to solver labels, checking for any unmapped labels
        self.solver_cell_labels = np.zeros_like(cell_labels)
        for region_label, solver_label in self.region_label_to_solver_label.items():
            self.solver_cell_labels[cell_labels == region_label] = solver_label
        if np.any(self.solver_cell_labels == 0):
            missing_labels = sorted({int(label) for label in np.unique(cell_labels[self.solver_cell_labels == 0])})
            self.loggers.errors(
                f"Mesh mapping produced tetrahedra with no solver label mapping for region labels {missing_labels}"
            )

        # Logging
        unlabelled = int(np.sum(cell_labels == 0))
        with open(os.path.join(self.log_dir, "labelled_cells_counts.txt"), "w") as f:
            f.write(f"Total global cells: {n_cells:,}\n")
            f.write(f"Unlabelled cells:   {unlabelled:,} ({unlabelled / n_cells:.2%})\n")
            for candidate in self.mapping_candidates:
                region_name = candidate["region"]
                region_label = candidate["region_label"]
                count = int(np.sum(cell_labels == region_label))
                f.write(f"{region_name:20s} {count:,}\n")

    def create_mesh_with_labels(self, labels: np.ndarray, output_path: str):
        """
        Create a VTK mesh with attached ROI labels for visualisation.

        Parameters:
        ---
        labels (np.ndarray) : Array of region labels for each tetrahedral cell.
        output_path (str)   : Path to save the output VTK file.
        """
        # Find the tetrahedral cell block index in the original mesh data
        tetra_block_index = None
        for block_index, cell_block in enumerate(self.global_mesh_data.cells):
            if cell_block.type in {"tetra", "tetra10"}:
                tetra_block_index = block_index
                break

        if tetra_block_index is None:
            self.loggers.errors(
                f"No supported cell block ['tetra', 'tetra10'] found in mesh file {self.global_mesh}"
            )

        # Create cell data array with labels for the tetrahedral block and zeros for other blocks
        cell_data = []
        for block_index, cell_block in enumerate(self.global_mesh_data.cells):
            if block_index == tetra_block_index:
                if len(labels) != len(cell_block.data):
                    self.loggers.errors("Cell label count does not match tetrahedral cell count")
                cell_data.append(np.asarray(labels, dtype=np.int64))
            else:
                cell_data.append(np.zeros(len(cell_block.data), dtype=np.int64))

        self.global_mesh_data.cell_data["ROI"] = cell_data

        # Write the mesh with labels to file
        stderr_orig = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            meshio.write(output_path, self.global_mesh_data, file_format="vtk", binary=False)
        finally:
            sys.stderr.close()
            sys.stderr = stderr_orig

    def map_scalar_to_tetra(self, nii_fpath: str, scalar_type: str):
        """
        Map a scalar field from a NIfTI volume onto the tetrahedral mesh.

        Parameters:
        ---
        nii_fpath (str)      : Path to the scalar NIfTI volume.
        scalar_type (str)    : Scalar type name for output file naming.
        """
        # Load data from NIfTI
        nii = nib.load(nii_fpath)
        img = nii.get_fdata()
        dim_x, dim_y, dim_z = img.shape
        affine = nii.affine
        inv_affine = np.linalg.inv(affine)

        mask_nonzero = img != 0
        average_val = img[mask_nonzero].mean()

        # Compute voxel coordinates of tetrahedral centroids and map scalar values
        coords = self.global_mesh_node_coords[self.global_mesh_tetra_indices, :3]
        centroids = coords.mean(axis=1)

        # Convert centroids to homogeneous coordinates for affine transformation
        homog_centroids = np.c_[centroids, np.ones(len(centroids))]
        voxel_coords = (inv_affine @ homog_centroids.T).T[:, :3]
        voxel_coords = np.round(voxel_coords).astype(int)

        # Clip voxel coordinates to be within image bounds
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, dim_x - 1)
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, dim_y - 1)
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, dim_z - 1)

        # Map scalar values to tetrahedra
        tetra_vals = img[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]

        # For any zero-valued voxels, compute the average of the 3x3x3 neighbourhood
        zero_indices = np.where(tetra_vals == 0)[0]
        for index in zero_indices:
            x, y, z = voxel_coords[index]
            xs = np.clip(np.arange(x - 2, x + 1), 0, dim_x - 1)
            ys = np.clip(np.arange(y - 2, y + 1), 0, dim_y - 1)
            zs = np.clip(np.arange(z - 2, z + 1), 0, dim_z - 1)
            cube = img[np.ix_(xs, ys, zs)]
            cube_flat = cube.flatten()
            mask_nonzero_cube = cube_flat != 0
            n_nonzero = np.sum(mask_nonzero_cube)
            n_zero = cube_flat.size - n_nonzero

            if n_nonzero == 0:
                tetra_vals[index] = average_val
            else:
                tetra_vals[index] = (
                    cube_flat[mask_nonzero_cube].mean() * n_nonzero + n_zero * average_val
                ) / cube_flat.size

        # Write scalar values to output file
        output_file = os.path.join(self.output_dir, f"{scalar_type}_map.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for cell_id, value in enumerate(np.asarray(tetra_vals, dtype=float), start=1):
                f.write(f"{cell_id} {value:f}\n")

        # Check output file was created
        if not os.path.isfile(output_file):
            self.loggers.errors(f"{scalar_type} scalar map file not produced - {output_file}")

    def run_mapping(self):
        """
        Run mesh mapping.
        """
        self.loggers.plugin_log("Mapping regional labels onto global mesh")

        self.interim_dir = os.path.join(self.interim_dir, "mesh_mapping")
        os.makedirs(self.interim_dir, exist_ok=True)

        self.loggers.verbose_log("Preparing mesh information inputs")
        self.prepare_mesh_info_inputs()

        self.loggers.verbose_log("Classifying tetrahedra into regions")
        self.classify_tetrahedra()

        self.loggers.verbose_log("Finalising cell labels for detailed and solver outputs")
        self.finalise_cell_labels()

        self.loggers.verbose_log("Creating VTK meshes with attached region and solver labels for visualisation")
        self.create_mesh_with_labels(self.region_cell_labels, os.path.join(self.interim_dir, "global_with_region_labels.vtk"))
        self.create_mesh_with_labels(self.solver_cell_labels, os.path.join(self.interim_dir, "global_with_solver_labels.vtk"))

        self.loggers.verbose_log("Writing region label file")
        self.labels_file = os.path.join(self.output_dir, "labels", "labels.txt")
        os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)
        with open(self.labels_file, "w") as f:
            for cell_id, label in enumerate(np.asarray(self.region_cell_labels, dtype=int), start=1):
                f.write(f"{cell_id} {label}\n")

        self.loggers.verbose_log("Writing solver label file")
        self.solver_labels_file = os.path.join(self.output_dir, "labels", "solver_labels.txt")
        with open(self.solver_labels_file, "w") as f:
            for cell_id, label in enumerate(np.asarray(self.solver_cell_labels, dtype=int), start=1):
                f.write(f"{cell_id} {label}\n")
