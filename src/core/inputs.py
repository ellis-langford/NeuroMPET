# Imports
import os
import sys
import shutil
import glob
import nibabel as nib

class Inputs(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributes are present
        to_inherit = [
            "utils",
            "loggers",
            "parameters",
            "base_dir",
            "log_dir",
            "input_dir",
            "interim_dir",
            "output_dir",
            "synthseg_env",
            "region_definitions",
        ]

        # Inherit required attributes from pipeline object
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def check_input_dir(self, input_flag: str, allow_file: bool = False):
        """
        Check and clean directory inputs.

        Parameters:
        ---
        input_flag (str)    : Command line flag.
        allow_file (bool)   : If True, a non-directory path is allowed and returns is_dir=False.
        """
        is_dir = False

        # Check if required input has been provided
        filepath = self.parameters[input_flag]
        if not filepath:
            self.loggers.errors(f"Required input --{input_flag} has not been provided")

        # Check if input is valid
        elif not os.path.isdir(filepath):
            if allow_file:
                return is_dir, filepath
            self.loggers.errors(f"Input path provided with --{input_flag} does not exist or is not a directory: {filepath}")

        # Check if directory is empty
        elif len(os.listdir(filepath)) == 0:
            self.loggers.errors(f"Input path provided with --{input_flag} points to an empty directory")

        # Inputs all ok
        else:
            is_dir = True
            return is_dir, filepath

    def check_input_files(self, input_flag: str, ext: str = ".nii.gz"):
        """
        Check and clean file inputs.
        
        Parameters:
        ---
        input_flag (str)   : Command line flag.
        ext (str)          : File extension.
        """
        # Check if required input has been provided
        filepaths = self.parameters[input_flag]
        if not filepaths:
            self.loggers.errors(f"Required input --{input_flag} has not been provided")

        # Check comma-separated list of files
        if "," in filepaths:
            # Check individual paths
            paths = filepaths.split(",")
            for path in paths:
                if not os.path.isfile(path):
                    self.loggers.errors(f"Input path provided with --{input_flag} does not exist: {path}")
            return paths

        else:
            # Check if filepath is valid
            if not os.path.isfile(filepaths):
                self.loggers.errors(f"Input path provided with --{input_flag} does not exist: {filepaths}")

            # Check if filepath has the correct extension
            elif not filepaths.endswith(ext):
                self.loggers.errors(f"Input path provided with --{input_flag} has the wrong extension: "
                                    f"expected {ext}, actual {filepaths}")
            else:
                return filepaths

    def define_paths(self):
        """
        Define all working paths used by the pipeline input layer.
        """
        # Input image & atlas
        self.input_im = os.path.join(self.input_dir, "image.nii.gz")
        self.atlas = os.path.join(self.input_dir, "atlas.nii.gz")

        # Segmentations
        self.segmentation_dir = (
            os.path.join(self.output_dir, "segmentations")
            if self.parameters["run_cortical_segmentation"]
            else os.path.join(self.input_dir, "segmentations")
        )
        self.global_mask_file = os.path.join(self.segmentation_dir, "global.nii.gz")

        # Surfaces
        self.surface_dir = (
            os.path.join(self.output_dir, "surfaces")
            if self.parameters["run_surface_generation"]
            else os.path.join(self.input_dir, "surfaces")
        )
        self.global_surface_file = os.path.join(self.surface_dir, "global.stl")

        # Meshes
        self.mesh_dir = (
            os.path.join(self.output_dir, "meshes")
            if self.parameters["run_mesh_generation"]
            else os.path.join(self.input_dir, "meshes")
        )
        self.mesh_file = os.path.join(self.mesh_dir, "global.vtk")

        # Modelling
        self.modelling_dir = (
            os.path.join(self.output_dir, "modelling")
            if self.parameters["run_modelling"]
            else os.path.join(self.input_dir, "modelling")
        )

        # Mesh mapping
        self.labels_dir = (
            os.path.join(self.output_dir, "labels")
            if self.parameters["run_mesh_mapping"]
            else os.path.join(self.input_dir, "labels")
        )
        self.labels_file = (os.path.join(self.labels_dir, "labels.txt"))
        self.solver_labels_file = (os.path.join(self.labels_dir, "solver_labels.txt"))

        # Boundary conditions
        self.bc_file = os.path.join(self.input_dir, "boundary_conditions_per_ml.csv")

    def validate_pipeline_steps(self):
        """
        Ensure enabled stages are consecutive with no disabled stage in the middle.
        """
        # Ordered stage flags used to enforce consecutive execution
        stage_flags = [
            ("run_preprocessing", self.parameters["run_preprocessing"]),
            ("run_registration", self.parameters["run_registration"]),
            ("run_cortical_segmentation", self.parameters["run_cortical_segmentation"]),
            ("run_surface_generation", self.parameters["run_surface_generation"]),
            ("run_mesh_generation", self.parameters["run_mesh_generation"]),
            ("run_mesh_mapping", self.parameters["run_mesh_mapping"]),
            ("run_modelling", self.parameters["run_modelling"]),
            ("run_results_processing", self.parameters["run_results_processing"]),
        ]

        # Identify active stage span
        true_idx = [i for i, (_, enabled) in enumerate(stage_flags) if enabled]
        if not true_idx:
            self.loggers.errors("At least one pipeline stage must be enabled")

        # Detect disabled stages inside active span
        first, last = true_idx[0], true_idx[-1]
        for idx in range(first, last + 1):
            if not stage_flags[idx][1]:
                prev_enabled = stage_flags[idx - 1][0]
                missing_stage = stage_flags[idx][0]
                next_enabled = stage_flags[idx + 1][0]
                self.loggers.errors(
                    f"Pipeline stages must be consecutive. "
                    f"Found disabled middle stage --{missing_stage} between "
                    f"--{prev_enabled} and --{next_enabled}"
                )

    def resolve_regions(self):
        """
        Create region name list based on supplied user inputs or region names.
        """
        # Get regions from parameters
        configured_regions = self.parameters.get("regions", [])

        # Build lookup dictionaries for region resolution
        label_to_region = {
            definition["region_label"]: region_name
            for region_name, definition in self.region_definitions.items()
            if "region_label" in definition
        }
        user_input_to_regions = {}
        for region_name, definition in self.region_definitions.items():
            user_input = definition.get("user_input")
            if user_input is not None:
                user_input_to_regions.setdefault(int(user_input), []).append(region_name)

        base_name_to_regions = {}
        for region_name in self.region_definitions:
            if region_name.endswith("_L") or region_name.endswith("_R"):
                base_name = region_name[:-2]
                base_name_to_regions.setdefault(base_name, []).append(region_name)

        def sort_regions(region_names):
            return sorted(
                region_names,
                key=lambda name: {"L": 0, "R": 1}.get(self.region_definitions[name].get("side"), 2),
            )

        single_region_names = {
            region_name
            for region_name in self.region_definitions
            if region_name not in base_name_to_regions
        }

        # Support comma-separated region lists as well as lists of strings/ints
        if isinstance(configured_regions, str):
            configured_regions = [region.strip() for region in configured_regions.split(",") if region.strip()]

        # Resolve user inputs to internal region names and validate all entries
        self.region_labels = []
        self.regions = []
        for region in configured_regions:
            resolved_regions = []
            region_label = None

            if isinstance(region, str) and region in base_name_to_regions:
                resolved_regions = sort_regions(base_name_to_regions[region])
                region_label = self.region_definitions.get(region, {}).get("region_label")
            elif isinstance(region, str) and region in self.region_definitions:
                if region in single_region_names:
                    resolved_regions = [region]
                else:
                    resolved_regions = sort_regions(base_name_to_regions.get(region, [region]))
                region_label = self.region_definitions[region].get("region_label")
            else:
                try:
                    numeric_region = int(region)
                except (TypeError, ValueError):
                    self.loggers.errors(f"Unknown region entry: {region}")

                if numeric_region in user_input_to_regions:
                    side_regions = [
                        region_name
                        for region_name in user_input_to_regions[numeric_region]
                        if self.region_definitions[region_name].get("side") in ("L", "R")
                    ]
                    resolved_regions = sort_regions(side_regions or user_input_to_regions[numeric_region])
                    region_label = numeric_region
                elif numeric_region in label_to_region:
                    resolved_regions = [label_to_region[numeric_region]]
                    region_label = numeric_region
                else:
                    self.loggers.errors(f"Unknown region input or label: {numeric_region}")

            if region_label is not None:
                self.region_labels.append(region_label)

            for region_name in resolved_regions:
                if region_name not in self.regions:
                    self.regions.append(region_name)

        required_solver_regions = [
            "cerebrumGM_L",
            "cerebrumGM_R",
            "cerebrumWM_L",
            "cerebrumWM_R",
            "cerebellumGM_L",
            "cerebellumGM_R",
            "cerebellumWM_L",
            "cerebellumWM_R",
            "brainstem_L",
            "brainstem_R",
        ]
        for region_name in required_solver_regions:
            if region_name not in self.region_definitions:
                self.loggers.errors(f"Required solver region is not defined: {region_name}")
            if region_name not in self.regions:
                self.regions.append(region_name)

        self.parameters["regions"] = self.regions

    def ensure_required_prepared_inputs(self):
        """
        Verify required external inputs were staged for the selected start stage.
        """
        # Starting at surface generation requires prepared segmentations
        if self.parameters["run_surface_generation"] and not self.parameters["run_cortical_segmentation"]:
            if not glob.glob(os.path.join(self.segmentation_dir, "*.nii*")):
                self.loggers.errors(
                    f"Missing prepared segmentation files in {self.segmentation_dir}"
                )

        # Starting at mesh generation requires prepared surfaces
        if self.parameters["run_mesh_generation"] and not self.parameters["run_surface_generation"]:
            if not glob.glob(os.path.join(self.surface_dir, "*.stl")):
                self.loggers.errors(
                    f"Missing prepared surface files in {self.surface_dir}"
                )

        # Starting at mesh mapping requires a prepared global mesh and surfaces
        if self.parameters["run_mesh_mapping"] and not self.parameters["run_mesh_generation"]:
            if not os.path.isfile(self.mesh_file):
                self.loggers.errors(
                    f"Missing prepared global mesh file in {self.mesh_file}"
                )
            if not glob.glob(os.path.join(self.surface_dir, "*.stl")):
                self.loggers.errors(
                    f"Missing prepared surface files in {self.surface_dir}"
                )

        # Starting at modelling requires prepared mesh/surfaces/labels
        if self.parameters["run_modelling"] and not self.parameters["run_mesh_mapping"]:
            if not os.path.isfile(self.mesh_file):
                self.loggers.errors(
                    f"Missing prepared global mesh file in {self.mesh_file}"
                )
            for req_surface in ["global", "wholebrain", "ventricles"]:
                if not glob.glob(os.path.join(self.surface_dir, f"*{req_surface}*.stl")):
                    self.loggers.errors(
                        f"Missing required prepared surface (*{req_surface}*.stl) in {self.surface_dir}"
                    )
            if not os.path.isfile(self.solver_labels_file):
                self.loggers.errors(
                    f"Missing prepared solver label file in {self.solver_labels_file}"
                )
            if not os.path.isfile(self.bc_file):
                self.loggers.errors(
                    f"Missing prepared boundary condition file in {self.bc_file}"
                )

        # Starting at results processing requires prepared modelling outputs
        if self.parameters["run_results_processing"] and not self.parameters["run_modelling"]:
            timestep = int(self.parameters["results_timestep"])
            metrics_file = os.path.join(self.modelling_dir, f"outputs_{timestep}.vtu")
            if not os.path.isfile(metrics_file):
                self.loggers.errors(
                    f"Missing prepared modelling output file in {metrics_file}"
                )
            if not os.path.isfile(self.input_im):
                self.loggers.errors(
                    f"Missing prepared reference image file in {self.input_im}"
                )
            global_matches = glob.glob(os.path.join(self.segmentation_dir, "*global*.nii*"))
            wholebrain_matches = glob.glob(os.path.join(self.segmentation_dir, "*wholebrain*.nii*"))
            if len(global_matches) > 1:
                self.loggers.errors(
                    f"Multiple global segmentation masks matched in {self.segmentation_dir}: {global_matches}"
                )
            if len(wholebrain_matches) > 1:
                self.loggers.errors(
                    f"Multiple wholebrain segmentation masks matched in {self.segmentation_dir}: {wholebrain_matches}"
                )
            if global_matches:
                self.global_mask_file = global_matches[0]
            elif wholebrain_matches:
                self.global_mask_file = wholebrain_matches[0]
            elif self.parameters["run_surface_generation"] and not self.parameters["run_cortical_segmentation"]:
                self.loggers.errors(
                    "A global or wholebrain segmentation must be provided with --segmentations"
                )
            else:
                self.loggers.errors(
                    "Results processing requires a staged global or wholebrain mask for voxel mapping"
                )
            if not os.path.isfile(self.labels_file):
                self.loggers.errors(
                    f"Missing prepared detailed region label file in {self.labels_file}"
                )
            input_shape = nib.load(self.input_im).shape[:3]
            global_mask_shape = nib.load(self.global_mask_file).shape[:3]
            if input_shape != global_mask_shape:
                self.loggers.errors(
                    f"Reference image shape {input_shape} does not match global mask shape {global_mask_shape}. "
                    f"When starting later in the pipeline, --input_im must match the space of the staged mask."
                )
            
    def stage_files_from_flag(self, input_flag: str, output_dir: str, ext: str = ".nii.gz", 
                              single_file_only: bool = False, output_name: str = None):
        """
        Stage inputs from a parameter flag into an output directory.
        """
        # Resolve user-provided input value
        input_value = self.parameters[input_flag]

        # Copy full directory inputs directly
        if os.path.isdir(input_value):
            _, input_dir = self.check_input_dir(input_flag)
            shutil.copytree(input_dir, output_dir)
            return

        # Validate and copy file inputs
        files = self.check_input_files(input_flag, ext=ext)
        if isinstance(files, list) and single_file_only:
            self.loggers.errors(f"Input --{input_flag} must be a single file or a directory")
        os.makedirs(output_dir, exist_ok=True)

        # Copy files to output directory, optionally renaming single file inputs
        if isinstance(files, list):
            for fpath in files:
                shutil.copy(fpath, output_dir)
        else:
            target_name = output_name if output_name else os.path.basename(files)
            shutil.copy(files, os.path.join(output_dir, target_name))

    def get_required_external_inputs(self):
        """
        Build a list of required external inputs to stage based on enabled pipeline steps.

        Returns:
        ---
        list : A list of tuples defining required inputs to stage.
        """
        required = []

        # Image and atlas inputs for preprocessing/registration/segmentation starts
        if any([
            self.parameters["run_preprocessing"],
            self.parameters["run_registration"],
            self.parameters["run_cortical_segmentation"],
        ]):
            required.append(("single", "input_im", self.input_im, ".nii.gz", True, None))

        # Atlas input is only required when registration is enabled
        if self.parameters["run_registration"]:
            required.append(("single", "input_atlas", self.atlas, ".nii.gz", False, None))

        # Results processing requires a reference image if earlier image stages are not running
        if self.parameters["run_results_processing"] and not any([
            self.parameters["run_preprocessing"],
            self.parameters["run_registration"],
            self.parameters["run_cortical_segmentation"],
        ]):
            required.append(("single", "input_im", self.input_im, ".nii.gz", True, None))

        # Starting at surface generation requires segmentation inputs
        if self.parameters["run_surface_generation"] and not self.parameters["run_cortical_segmentation"]:
            required.append(("multi", "segmentations", self.segmentation_dir, ".nii.gz", False, None))

        # Starting at mesh generation requires surface inputs
        if self.parameters["run_mesh_generation"] and not self.parameters["run_surface_generation"]:
            required.append(("multi", "surfaces", self.surface_dir, ".stl", False, None))

        # Starting at mesh mapping requires a global mesh + possibly surfaces
        if self.parameters["run_mesh_mapping"] and not self.parameters["run_mesh_generation"]:
            required.append(("single", "mesh", self.mesh_file, ".vtk", True, None))
            if not self.parameters["run_surface_generation"]:
                required.append(("multi", "surfaces", self.surface_dir, ".stl", False, None))

        # Starting at modelling (without mapping) requires mesh/surfaces/labels
        if self.parameters["run_modelling"] and not self.parameters["run_mesh_mapping"]:
            if not self.parameters["run_mesh_generation"]:
                required.append(("single", "mesh", self.mesh_file, ".vtk", True, None))
            if not self.parameters["run_surface_generation"]:
                required.append(("multi", "surfaces", self.surface_dir, ".stl", False, None))
            required.append(("multi", "solver_labels_file", os.path.dirname(self.solver_labels_file), ".txt", True, "solver_labels.txt"))
            if self.parameters["run_results_processing"]:
                required.append(("multi", "labels_file", os.path.dirname(self.labels_file), ".txt", True, "labels.txt"))

        # Starting at results processing requires modelling outputs
        if self.parameters["run_results_processing"] and not self.parameters["run_modelling"]:
            required.append(("multi", "modelling_outputs", self.modelling_dir, ".vtu", False, None))
            if not self.parameters["run_mesh_mapping"]:
                required.append(("multi", "labels_file", os.path.dirname(self.labels_file), ".txt", True, "labels.txt"))
            if self.parameters["register_to_mni"]:
                required.append(("single", "input_atlas", self.atlas, ".nii.gz", False, None))
        if self.parameters["run_results_processing"] and not self.parameters["run_cortical_segmentation"] and not self.parameters["run_surface_generation"]:
            required.append(("single", "global_mask", self.global_mask_file, ".nii.gz", True, "global.nii.gz"))

        # Modelling always requires a boundary condition file
        if self.parameters["run_modelling"]:
            required.append((
                "single",
                "bc_file",
                self.bc_file,
                ".csv",
                False,
                "/app/assets/boundary_conditions_per_ml.csv",
            ))

        # Deduplicate while preserving order
        unique = []
        seen = set()
        for item in required:
            key = (item[0], item[1], item[2])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def stage_required_external_inputs(self):
        """
        Stage all externally-required inputs from a requirements loop.
        """
        # Stage each requirement using single-file or multi-file flow
        for mode, input_flag, target, ext, required, extra in self.get_required_external_inputs():

            # Single-file staging flow
            if mode == "single":
                input_value = self.parameters.get(input_flag, "")
                if input_value:
                    source_file = self.check_input_files(input_flag, ext=ext)
                    if isinstance(source_file, list):
                        self.loggers.errors(f"Input --{input_flag} must be a single file")
                elif extra:
                    source_file = extra
                elif required:
                    self.loggers.errors(f"Required input --{input_flag} has not been provided")
                else:
                    continue

                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy(source_file, target)

            # Multi-file & directory staging flow
            else:
                self.stage_files_from_flag(
                    input_flag=input_flag,
                    output_dir=target,
                    ext=ext,
                    single_file_only=required,
                    output_name=extra,
                )

    def prepare_inputs(self):
        """
        Begin input preparation.
        """
        self.loggers.plugin_log("Preparing inputs")

        # Define core paths used by the input staging layer
        self.loggers.verbose_log("Defining paths")
        self.define_paths()

        # Ensure enabled pipeline stages are consecutive
        self.loggers.verbose_log("Validating pipeline stage configuration")
        self.validate_pipeline_steps()

        # Resolve configured regions once for downstream stages
        self.loggers.verbose_log("Resolving regions")
        self.resolve_regions()

        # Stage external inputs required by the selected pipeline start point
        self.loggers.verbose_log("Staging required external inputs")
        self.stage_required_external_inputs()

        # Verify staged inputs exist before running the pipeline
        self.loggers.verbose_log("Verifying required prepared inputs")
        self.ensure_required_prepared_inputs()
