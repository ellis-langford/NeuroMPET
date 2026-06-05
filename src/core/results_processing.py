# Imports
import glob
import os
import sys
import ants
import meshio
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


class ResultsProcessor(object):
    """Class setup"""
    metric_names = [
        "displacement",
        "volumetric_strain",
        "pressure_A",
        "pressure_C",
        "pressure_E",
        "pressure_V",
        "fluid_content_A",
        "fluid_content_C",
        "fluid_content_E",
        "fluid_content_V",
        "Darcy_velocity_A",
        "Darcy_velocity_C",
        "Darcy_velocity_E",
        "Darcy_velocity_V",
    ]

    def __init__(self, plugin_obj: object):
        # Check all expected attributes are present
        to_inherit = [
            "loggers",
            "parameters",
            "base_dir",
            "input_dir",
            "interim_dir",
            "output_dir",
            "log_dir",
            "segmentation_dir",
            "modelling_dir",
            "atlas",
            "regions",
            "region_definitions",
            "labels_file",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def get_tetra_cells(self, mesh: object) -> np.ndarray:
        """
        Extract tetrahedral cell connectivity from a mesh.

        Parameters:
        ---
        mesh   : Mesh object containing tetrahedral cells.

        Returns:
        ---
        np.ndarray : Tetrahedral cell connectivity array.
        """
        # Extract first supported tetrahedral cell block
        for cell_block in mesh.cells:
            if cell_block.type == "tetra":
                return np.asarray(cell_block.data, dtype=int)
            if cell_block.type == "tetra10":
                return np.asarray(cell_block.data[:, :4], dtype=int)
        self.loggers.errors("No tetrahedral cell block found in mesh")

    def get_point_or_cell_data(self, mesh: object, name: str, n_cells: int) -> tuple[np.ndarray, str]:
        """
        Extract a named field from mesh point or cell data.

        Parameters:
        ---
        mesh            : Mesh object containing result fields.
        name (str)      : Name of the field to extract.
        n_cells (int)   : Number of tetrahedral cells in the mesh.

        Returns:
        ---
        tuple : Field data array and whether it came from points or cells.
        """
        # Check point data first
        if name in mesh.point_data:
            return np.asarray(mesh.point_data[name]), "point"

        # Otherwise look for a matching tetrahedral cell-data array
        cell_data = mesh.cell_data.get(name)
        if cell_data:
            for values in cell_data:
                arr = np.asarray(values)
                if arr.shape[0] == n_cells:
                    return arr, "cell"

        self.loggers.errors(f"Field '{name}' not present in mesh data")

    def load_region_labels(self, n_cells: int) -> np.ndarray:
        """
        Load detailed tetrahedral region labels from file.

        Parameters:
        ---
        n_cells (int)   : Number of tetrahedral cells expected in the mesh.

        Returns:
        ---
        np.ndarray : Detailed region label per tetrahedron.
        """
        # Initialise label array and check the expected file exists
        labels = np.zeros(n_cells, dtype=int)
        if not os.path.isfile(self.labels_file):
            self.loggers.errors(f"Detailed region label file does not exist: {self.labels_file}")

        # Read one-indexed tetrahedral labels from file
        with open(self.labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                cell_id, region_label = parts
                idx = int(cell_id) - 1
                if idx < 0 or idx >= n_cells:
                    self.loggers.errors(
                        f"Detailed region label file index {cell_id} is out of bounds for {n_cells} cells"
                    )
                labels[idx] = int(region_label)

        # Ensure every tetrahedron received a detailed label
        if np.any(labels == 0):
            self.loggers.errors("Detailed region label file contains unlabelled tetrahedra")

        return labels

    def resolve_region_mask(self, region_labels: np.ndarray, region_name: str) -> np.ndarray:
        """
        Resolve the tetrahedral mask for a requested region.

        Parameters:
        ---
        region_labels       : Detailed region label per tetrahedron.
        region_name (str)   : Region name to resolve.

        Returns:
        ---
        np.ndarray : Boolean mask for tetrahedra belonging to the region.
        """
        # Global covers all labelled tetrahedra
        if region_name == "global":
            return region_labels > 0

        # Get direct region label from the region definitions
        region_definition = self.region_definitions[region_name]
        region_label = int(region_definition["region_label"])

        # Derived regions aggregate all detailed labels that share the same solver class
        if (
            region_definition.get("region_type") == "derived"
            and region_definition.get("solver_label") is not None
        ):
            solver_label = int(region_definition["solver_label"])
            matching_region_labels = [
                int(definition["region_label"])
                for definition in self.region_definitions.values()
                if definition.get("solver_label") == solver_label
                and definition.get("region_label") is not None
            ]
            return np.isin(region_labels, matching_region_labels)

        return region_labels == region_label

    def compute_tetra_volumes(self, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """
        Compute tetrahedral volumes from point coordinates and connectivity.

        Parameters:
        ---
        points   : Mesh point coordinates.
        cells    : Tetrahedral cell connectivity array.

        Returns:
        ---
        np.ndarray : Tetrahedral volumes.
        """
        # Compute tetrahedral volumes from the determinant formula
        p0 = points[cells[:, 0]]
        p1 = points[cells[:, 1]]
        p2 = points[cells[:, 2]]
        p3 = points[cells[:, 3]]
        return np.abs(np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    def compute_results(self, metrics_path: str, volume_weighted: bool = True) -> tuple[dict, list]:
        """
        Compute regional summary metrics from the final VTU output and detailed labels.
        """
        # Load solver output mesh and detailed tetrahedral labels
        metrics_mesh = meshio.read(metrics_path)

        points = np.asarray(metrics_mesh.points, dtype=float)
        cells = self.get_tetra_cells(metrics_mesh)
        n_cells = cells.shape[0]
        region_labels = self.load_region_labels(n_cells)
        volumes = self.compute_tetra_volumes(points, cells)

        # Build region/hemisphere masks for all requested outputs
        result_specs = [
            {
                "region": "global",
                "hemisphere": "both",
                "mask": region_labels > 0,
            }
        ]

        region_label_to_name = {
            definition["region_label"]: region_name
            for region_name, definition in self.region_definitions.items()
            if "region_label" in definition
        }

        left_mask = np.zeros_like(region_labels, dtype=bool)
        right_mask = np.zeros_like(region_labels, dtype=bool)
        for region_label, region_name in region_label_to_name.items():
            if region_name.endswith("_L"):
                left_mask |= region_labels == int(region_label)
            elif region_name.endswith("_R"):
                right_mask |= region_labels == int(region_label)

        result_specs.append({"region": "global", "hemisphere": "left", "mask": left_mask})
        result_specs.append({"region": "global", "hemisphere": "right", "mask": right_mask})

        base_sides = {}
        for region_name in self.regions:
            if region_name == "global":
                continue

            if region_name.endswith("_L"):
                base_name = region_name[:-2]
                hemisphere = "left"
                side_code = "L"
            elif region_name.endswith("_R"):
                base_name = region_name[:-2]
                hemisphere = "right"
                side_code = "R"
            else:
                base_name = region_name
                hemisphere = "both"
                side_code = None

            mask = self.resolve_region_mask(region_labels, region_name)
            result_specs.append(
                {
                    "region": base_name,
                    "hemisphere": hemisphere,
                    "mask": mask,
                }
            )

            if side_code is not None:
                base_sides.setdefault(base_name, {})[side_code] = mask

        for base_name, sides in base_sides.items():
            if "L" in sides and "R" in sides:
                result_specs.append(
                    {
                        "region": base_name,
                        "hemisphere": "both",
                        "mask": sides["L"] | sides["R"],
                    }
                )

        # Add composite reporting masks from final detailed labels. These overwrite
        # any same-named specs created above from direct region definitions.
        composite_solver_labels = {
            "cerebrumGM": {
                "left": (1,),
                "right": (2,),
            },
            "cerebrum": {
                "left": (1, 3),
                "right": (2, 4),
            },
            "cerebellum": {
                "left": (7, 9),
                "right": (8, 10),
            },
        }
        result_specs = [
            spec
            for spec in result_specs
            if spec["region"] not in composite_solver_labels
        ]

        def labels_for_solver_labels(solver_labels: tuple[int, ...]) -> list[int]:
            return [
                int(definition["region_label"])
                for definition in self.region_definitions.values()
                if definition.get("region_label") is not None
                and definition.get("solver_label") is not None
                and int(definition["solver_label"]) in solver_labels
            ]

        for composite_region, hemisphere_solver_labels in composite_solver_labels.items():
            left_mask = np.isin(
                region_labels,
                labels_for_solver_labels(hemisphere_solver_labels["left"]),
            )
            right_mask = np.isin(
                region_labels,
                labels_for_solver_labels(hemisphere_solver_labels["right"]),
            )
            result_specs.extend(
                [
                    {"region": composite_region, "hemisphere": "left", "mask": left_mask},
                    {"region": composite_region, "hemisphere": "right", "mask": right_mask},
                    {"region": composite_region, "hemisphere": "both", "mask": left_mask | right_mask},
                ]
            )

        # Compute each metric for each requested region/hemisphere combination
        results = {}
        for metric in self.metric_names:
            data, _ = self.get_point_or_cell_data(
                metrics_mesh,
                metric,
                n_cells=n_cells,
            )
            data = np.asarray(data)

            # Convert point-based fields to cell values if needed
            if data.shape[0] == n_cells:
                cell_values = data
            else:
                cell_values = (
                    data[cells[:, 0]]
                    + data[cells[:, 1]]
                    + data[cells[:, 2]]
                    + data[cells[:, 3]]
                ) / 4.0

            # Convert vector fields to magnitudes and apply unit conversions
            if cell_values.ndim == 2:
                cell_values = np.linalg.norm(cell_values, axis=1)
                if metric == "Darcy_velocity_C":
                    cell_values = cell_values * 6e5
                elif metric.startswith("Darcy_velocity"):
                    cell_values = cell_values * 10e6

            cell_values = np.asarray(cell_values).reshape(-1)
            for spec in result_specs:
                key = f"{metric}|{spec['region']}|{spec['hemisphere']}"
                if spec["mask"] is None or not np.any(spec["mask"]):
                    results[key] = np.nan
                    continue

                if volume_weighted:
                    masked_volumes = volumes[spec["mask"]]
                    if masked_volumes.size == 0 or masked_volumes.sum() == 0:
                        results[key] = np.nan
                    else:
                        results[key] = float(
                            np.sum(cell_values[spec["mask"]] * masked_volumes)
                            / masked_volumes.sum()
                        )
                else:
                    results[key] = float(np.mean(cell_values[spec["mask"]]))

        return results, result_specs

    def build_results_dataframe(self, results: dict, result_specs: list) -> pd.DataFrame:
        """
        Build and validate the long-format regional results dataframe.
        """
        # Rebuild the expected long-format result keys from the computed specs
        expected_keys = [
            f"{metric}|{spec['region']}|{spec['hemisphere']}"
            for metric in self.metric_names
            for spec in result_specs
        ]
        # Fail early if any expected results are missing or invalid
        missing_columns = [key for key in expected_keys if key not in results]
        if missing_columns:
            self.loggers.errors(
                f"Results dataframe is missing expected metrics: {missing_columns}"
            )

        missing_values = [key for key in expected_keys if pd.isna(results[key])]
        if missing_values:
            self.loggers.errors(
                f"Results dataframe contains missing metric values for columns: {missing_values}"
            )

        # Convert the keyed results dictionary into a long-format dataframe
        rows = []
        for key in expected_keys:
            metric, region, hemisphere = key.split("|")
            rows.append(
                {
                    "Metric": metric,
                    "Region": region,
                    "Hemisphere": hemisphere,
                    "Value": float(results[key]),
                }
            )

        results_df = pd.DataFrame(
            rows,
            columns=["Metric", "Region", "Hemisphere", "Value"],
        )
        return results_df

    def build_summary_dataframe(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a summary dataframe containing global bilateral results for all metrics.
        """
        # Keep only the global bilateral rows for the summary output
        summary_df = results_df.loc[
            (results_df["Region"] == "global")
            & (results_df["Hemisphere"] == "both")
        ].copy()
        summary_df = summary_df.loc[:, ["Metric", "Value"]]
        return summary_df

    def create_nifti_maps(
        self,
        metrics_path: str,
        ref_nii_path: str,
        outdir: str,
        mask_nii_path: str,
        regions="All",
        max_dist_mm: float = 3.0,
        fill_value: float = 0.0,
    ) -> tuple[str, str]:
        """
        Create CBF and CSF NIfTI maps from the final VTU output.
        """
        # Ensure output directory exists and load the final solver output
        os.makedirs(outdir, exist_ok=True)

        metrics_mesh = meshio.read(metrics_path)

        # Load mesh geometry and detailed tetrahedral labels
        points_mm = np.asarray(metrics_mesh.points, dtype=float) * 1000.0
        cells = self.get_tetra_cells(metrics_mesh)
        n_points = points_mm.shape[0]
        n_cells = cells.shape[0]
        region_labels = self.load_region_labels(n_cells)

        # Select tetrahedra contributing to the requested regional map
        region_label_to_name = {
            definition["region_label"]: region_name
            for region_name, definition in self.region_definitions.items()
            if "region_label" in definition
        }

        if regions == "All":
            cell_mask = np.ones(n_cells, dtype=bool)
            cbf_name = "CBF.nii.gz"
            csf_name = "CSF.nii.gz"
        else:
            if isinstance(regions, str):
                regions = [regions]
            region_masks = [
                self.resolve_region_mask(
                    region_labels,
                    region_label_to_name.get(int(region), region) if not isinstance(region, str) else region,
                )
                for region in regions
            ]
            cell_mask = np.any(np.column_stack(region_masks), axis=1)
            cbf_name = "CBF_WM.nii.gz"
            csf_name = "CSF_WM.nii.gz"

        # Restrict the mapping to the selected tetrahedra
        selected_cells = cells[cell_mask]
        if selected_cells.shape[0] == 0:
            self.loggers.errors("No tetrahedra selected for NIfTI map generation")

        # Compute nodal CBF and CSF from volume-weighted tetrahedral vector magnitudes
        volumes = self.compute_tetra_volumes(points_mm, selected_cells)
        darcy_c, _ = self.get_point_or_cell_data(
            metrics_mesh,
            "Darcy_velocity_C",
            n_cells=n_cells,
        )
        darcy_e, _ = self.get_point_or_cell_data(
            metrics_mesh,
            "Darcy_velocity_E",
            n_cells=n_cells,
        )
        darcy_c = np.asarray(darcy_c, dtype=float)
        darcy_e = np.asarray(darcy_e, dtype=float)

        cbf_cell_vec = (
            darcy_c[selected_cells[:, 0]]
            + darcy_c[selected_cells[:, 1]]
            + darcy_c[selected_cells[:, 2]]
            + darcy_c[selected_cells[:, 3]]
        ) / 4.0
        cbf_cell_mag = np.linalg.norm(cbf_cell_vec, axis=1)
        cbf_num = np.zeros(n_points, dtype=float)
        cbf_den = np.zeros(n_points, dtype=float)
        for j in range(4):
            idx = selected_cells[:, j]
            np.add.at(cbf_num, idx, cbf_cell_mag * volumes)
            np.add.at(cbf_den, idx, volumes)
        cbf_nodes = np.divide(
            cbf_num,
            cbf_den,
            out=np.full(n_points, np.nan),
            where=cbf_den > 0,
        ) * 6e5

        csf_cell_vec = (
            darcy_e[selected_cells[:, 0]]
            + darcy_e[selected_cells[:, 1]]
            + darcy_e[selected_cells[:, 2]]
            + darcy_e[selected_cells[:, 3]]
        ) / 4.0
        csf_cell_mag = np.linalg.norm(csf_cell_vec, axis=1)
        csf_num = np.zeros(n_points, dtype=float)
        csf_den = np.zeros(n_points, dtype=float)
        for j in range(4):
            idx = selected_cells[:, j]
            np.add.at(csf_num, idx, csf_cell_mag * volumes)
            np.add.at(csf_den, idx, volumes)
        csf_nodes = np.divide(
            csf_num,
            csf_den,
            out=np.full(n_points, np.nan),
            where=csf_den > 0,
        ) * 10e6

        # Load the reference image and global mask used for voxel mapping
        ref_img = nib.load(ref_nii_path)
        ref_aff = ref_img.affine
        shape = ref_img.shape[:3]

        if not mask_nii_path or not os.path.isfile(mask_nii_path):
            self.loggers.errors("No global mask available for results voxel mapping")

        mask_img = nib.load(mask_nii_path)
        mask = mask_img.get_fdata().astype(bool)
        if mask.shape[:3] != shape:
            self.loggers.errors("Global mask shape does not match reference image shape")

        # Convert masked voxel indices to world coordinates for nearest-node sampling
        ijk = np.indices(shape).reshape(3, -1).T
        ijk_masked = ijk[mask.ravel()]
        ijk_h = np.hstack([ijk_masked, np.ones((ijk_masked.shape[0], 1))])
        xyz_mm = (ref_aff @ ijk_h.T).T[:, :3]

        # Map voxel centres to the nearest solver mesh node
        tree = KDTree(points_mm)
        dist_mm, nn = tree.query(xyz_mm, k=1)

        # Map node values onto the masked voxel grid
        cbf_out = np.full(np.prod(shape), fill_value, dtype=np.float32)
        cbf_sampled = cbf_nodes[nn]
        cbf_sampled = np.where(np.isfinite(cbf_sampled), cbf_sampled, fill_value)
        if max_dist_mm is not None:
            cbf_sampled = np.where(dist_mm <= max_dist_mm, cbf_sampled, fill_value)
        cbf_out[np.flatnonzero(mask.ravel())] = cbf_sampled.astype(np.float32)

        csf_out = np.full(np.prod(shape), fill_value, dtype=np.float32)
        csf_sampled = csf_nodes[nn]
        csf_sampled = np.where(np.isfinite(csf_sampled), csf_sampled, fill_value)
        if max_dist_mm is not None:
            csf_sampled = np.where(dist_mm <= max_dist_mm, csf_sampled, fill_value)
        csf_out[np.flatnonzero(mask.ravel())] = csf_sampled.astype(np.float32)

        # Save CBF and CSF maps
        nib.save(
            nib.Nifti1Image(cbf_out.reshape(shape), ref_aff, ref_img.header),
            os.path.join(outdir, cbf_name),
        )
        nib.save(
            nib.Nifti1Image(csf_out.reshape(shape), ref_aff, ref_img.header),
            os.path.join(outdir, csf_name),
        )
        return (
            os.path.join(outdir, cbf_name),
            os.path.join(outdir, csf_name),
        )

    def resolve_reference_nifti(self) -> str:
        """
        Resolve the reference NIfTI image for result voxel mapping.

        Returns:
        ---
        str : File path to the reference NIfTI image.
        """
        # Prefer the final output image, then staged input image, then staged segmentations
        candidates = [
            os.path.join(self.output_dir, "image.nii.gz"),
            os.path.join(self.input_dir, "image.nii.gz"),
        ]
        if self.segmentation_dir:
            candidates.extend(glob.glob(os.path.join(self.segmentation_dir, "*wholebrain*.nii*")))
            candidates.extend(glob.glob(os.path.join(self.segmentation_dir, "*.nii*")))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        self.loggers.errors("Unable to locate a reference NIfTI image for result mapping")

    def resolve_mask_nifti(self) -> str:
        """
        Resolve the global mask NIfTI used for result voxel mapping.

        Returns:
        ---
        str : File path to the global mask NIfTI.
        """
        # Prefer the staged global mask, then fall back to wholebrain if needed
        if not self.segmentation_dir:
            return None
        global_matches = glob.glob(os.path.join(self.segmentation_dir, "*global*.nii*"))
        wholebrain_matches = glob.glob(os.path.join(self.segmentation_dir, "*wholebrain*.nii*"))
        if global_matches:
            return global_matches[0]
        if wholebrain_matches:
            return wholebrain_matches[0]
        return None

    def resolve_solver_output(self) -> str:
        """
        Resolve the final solver output VTU for results processing.

        Returns:
        ---
        str : File path to the solver output VTU.
        """
        # Resolve the requested solver output timestep
        timestep = int(self.parameters["results_timestep"])
        metrics_path = os.path.join(self.modelling_dir, f"outputs_{timestep}.vtu")
        if not os.path.isfile(metrics_path):
            self.loggers.errors(
                f"Unable to locate solver timestep output file {metrics_path}"
            )
        return metrics_path

    def register_results_to_mni(self, ref_nii_path: str, cbf_path: str, csf_path: str, output_dir: str):
        """
        Register the final image and result NIfTIs to the MNI atlas.
        """
        # Load atlas and reference image for registration to MNI space
        atlas_path = self.atlas
        reg_type = self.parameters["reg_type"]

        fixed_image = ants.image_read(atlas_path)
        moving_image = ants.image_read(ref_nii_path)
        self.loggers.plugin_log("Registering final image and result NIfTIs to MNI space")
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=reg_type,
        )
        transform_list = registration["fwdtransforms"]

        # Save the registered image and apply the same transform to the result maps
        mni_image_path = os.path.join(output_dir, "image_mni.nii.gz")
        ants.image_write(registration["warpedmovout"], mni_image_path)

        for nifti_path in [cbf_path, csf_path]:
            transformed = ants.apply_transforms(
                fixed=fixed_image,
                moving=ants.image_read(nifti_path),
                transformlist=transform_list,
                interpolator="linear",
            )
            ants.image_write(transformed, nifti_path)

        if not os.path.isfile(mni_image_path):
            self.loggers.errors("Registered MNI reference image was not produced")

    def run_results_processing(self):
        """
        Run final solver-output post-processing and generate summary files.
        """
        # Resolve the required solver output, reference image, and global mask
        metrics_path = self.resolve_solver_output()
        ref_nii_path = self.resolve_reference_nifti()
        mask_nii_path = self.resolve_mask_nifti()

        # Ensure the main output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        results_plot_dir = os.path.join(self.output_dir, "results_plots")
        os.makedirs(results_plot_dir, exist_ok=True)

        # Build tabular results outputs
        self.loggers.plugin_log("Computing results metrics")
        results, result_specs = self.compute_results(
            metrics_path,
            volume_weighted=bool(self.parameters["volume_weighted_results"]),
        )
        results_df = self.build_results_dataframe(results, result_specs)
        summary_df = self.build_summary_dataframe(results_df)
        results_df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        summary_df.to_csv(os.path.join(self.output_dir, "summary.csv"), index=False)

        # Create the subject NIfTI result maps
        self.loggers.plugin_log("Creating CBF and CSF NIfTI maps")
        cbf_path, csf_path = self.create_nifti_maps(
            metrics_path=metrics_path,
            ref_nii_path=ref_nii_path,
            outdir=results_plot_dir,
            mask_nii_path=mask_nii_path,
            regions="All",
            max_dist_mm=float(self.parameters["results_max_dist_mm"]),
        )

        # Optionally register the final result maps to MNI space
        if self.parameters["register_to_mni"]:
            self.register_results_to_mni(
                ref_nii_path=ref_nii_path,
                cbf_path=cbf_path,
                csf_path=csf_path,
                output_dir=results_plot_dir,
            )
