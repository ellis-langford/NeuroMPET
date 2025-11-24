# Imports
import os
import sys
import shutil
import glob

class Inputs(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["utils", "loggers", "parameters",
                      "base_dir", "log_dir", "input_dir", 
                      "interim_dir", "output_dir", "tmp_dir",
                      "freesurfer_env"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def check_input_dir(self, input_flag):
        """
        Helper function which checks and cleans directory inputs
        """
        # Check if required input has been provided
        if not self.parameters[input_flag]:
            self.loggers.errors(f"Required input --{input_flag} has not been provided")
        else:
            filepath = self.parameters[input_flag]

        # Check comma seperated list of files
        if "," in self.parameters[input_flag]:
            # Check individual paths
            paths = self.parameters[input_flag].split(",")
            for path in paths:
                if not os.path.isfile(path):
                    self.loggers.errors(f"Input path provided with --{input_flag} does not exist: {path}")
            # Inputs all ok
            is_dir = False
            return is_dir, paths
                    
        # Check directory
        else:
            # Check if input is valid
            if not os.path.isdir(filepath):
                    self.loggers.errors(f"Input path provided with --{input_flag} does not exist: {filepath}")
            # Check if directory is empty
            elif len(os.listdir(filepath)) == 0:
                self.loggers.errors(f"Input path provided with --{input_flag} points to an empty directory")
            # Inputs all ok
            else:
                is_dir = True
                return is_dir, filepath

    def check_input_file(self, input_flag, ext):
        """
        Helper function which checks and cleans file inputs
        """
        # Check if required input has been provided
        if not self.parameters[input_flag]:
            self.loggers.errors(f"Required input --{input_flag} has not been provided")
        else:
            filepath = self.parameters[input_flag]

        # Check if filepath is valid
        if not os.path.isfile(filepath):
            self.loggers.errors(f"Input path provided with --{input_flag} does not exist: {filepath}")
        # Check if filepath is the correct extension
        elif not filepath.endswith(ext):
            self.loggers.errors(f"Input path provided with --{input_flag} has the wrong extension: "
                                f"expected {ext}, actual {filepath}")
        else:
            return filepath
            
    def prepare_input_image(self):
        """
        Copy input image and atlas to working directory
        """
        # Prepare freesurfer-based inputs
        if not self.parameters["input_im"]:
            _, input_im = self.check_input_dir("freesurfer_outputs")
            shutil.copytree(input_im, self.fs_outputs)
            
         # Prepare input image
        else:
            input_im = self.check_input_file("input_im", ext=".nii.gz")
            self.input_im = os.path.join(self.input_dir, "image.nii.gz")
            shutil.copy(input_im, self.input_im)

        # Prepare atlas input
        if self.parameters["input_atlas"]:
            input_im = self.check_input_file("input_atlas", ext=".nii.gz")
            atlas_path = os.path.join(self.input_dir, "atlas.nii.gz")
            shutil.copy(input_im, atlas_path)   
            
    def prepare_seg_inputs(self):
        """
        Copy segmentation inputs to working directory
        """
        # Inputs provided
        if not self.parameters["run_cortical_segmentation"]:
            # Destination directory
            self.segmentation_dir = os.path.join(self.input_dir, "segmentations")

            # Check input
            is_dir, segmentation_dir = self.check_input_dir("segmentations")

            # Directory of segmentations
            if is_dir:
                shutil.copytree(segmentation_dir, self.segmentation_dir)
            # List of comma seperated segmentation paths
            else:
                os.makedirs(self.segmentation_dir)
                for path in segmentation_dir:
                    shutil.copy(path, self.segmentation_dir)

        # Inputs from prior processing step
        else:
            self.segmentation_dir = os.path.join(self.output_dir, "segmentations")

    def prepare_surface_inputs(self, fixed_surfaces=False):
        """
        Copy surface inputs to working directory
        """
        # Inputs provided
        if not self.parameters["run_surface_generation"]:
            # Destination directory
            self.surface_dir = os.path.join(self.input_dir, "surfaces")

            # Check input
            is_dir, surface_dir = self.check_input_dir("surfaces")

            # Directory of surfaces
            if is_dir:
                shutil.copytree(surface_dir, self.surface_dir)
            # List of comma seperated surface paths
            else:
                os.makedirs(self.surface_dir)
                for path in surface_dir:
                    shutil.copy(path, self.surface_dir)

        # Inputs from prior processing step
        else:
            # Fixed surface inputs from meshing stage
            if fixed_surfaces:
                self.surface_dir = os.path.join(self.output_dir, "meshes")
            # Before fix surface inputs from surface generation stage
            else:
                self.surface_dir = os.path.join(self.output_dir, "surfaces")

    def prepare_mesh_inputs(self):
        """
        Copy mesh inputs to working directory
        """
        # Inputs provided
        if not self.parameters["run_mesh_generation"]:
            # Destination directory
            self.mesh_dir = os.path.join(self.input_dir, "meshes")

            # Check input
            is_dir, mesh_dir = self.check_input_dir("meshes")

            # Directory of meshes
            if is_dir:
                shutil.copytree(mesh_dir, self.mesh_dir)
            # List of comma seperated mesh paths
            else:
                os.makedirs(self.mesh_dir)
                for path in mesh_dir:
                    shutil.copy(path, self.mesh_dir)

        # Inputs from prior processing step
        else:
            self.mesh_dir = os.path.join(self.output_dir, "meshes")
   
    def prepare_dwi_inputs(self):
        """
        Prepare diffusion weighted imaging inputs
        """
        # Destination directory
        self.dwi_dir = os.path.join(self.input_dir, "dwi_files")
        os.makedirs(self.dwi_dir, exist_ok=True)

        # Check input
        _, dwi_dir = self.check_input_dir("dwi_dir")

        # Copy to working directory
        try:
            shutil.copy(glob.glob(os.path.join(dwi_dir, "*tensor*.nii*"))[0], os.path.join(self.dwi_dir, f"dwi_tensor.nii.gz"))
            shutil.copy(glob.glob(os.path.join(dwi_dir, "*L1*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_L1.nii.gz"))
            shutil.copy(glob.glob(os.path.join(dwi_dir, "*L2*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_L2.nii.gz"))
            shutil.copy(glob.glob(os.path.join(dwi_dir, "*L3*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_L3.nii.gz"))
            shutil.copy(glob.glob(os.path.join(dwi_dir, "*FA*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_FA.nii.gz"))
            shutil.copy(glob.glob(os.path.join(dwi_dir, "*MD*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_MD.nii.gz"))
        except Exception as e:
            self.loggers.errors(f"Problem copying dwi_dir input {e}")

    def prepare_cbf_inputs(self):
        """
        Prepare cerebral blood flow imaging inputs
        """
        # Destination directory
        self.cbf_dir = os.path.join(self.input_dir, "cbf_files")
        os.makedirs(self.cbf_dir, exist_ok=True)

        # Check input
        _, cbf_dir = self.check_input_dir("cbf_dir")

        # Copy to working directory
        try:
            shutil.copy(glob.glob(os.path.join(cbf_dir, "*.nii.gz"))[0], os.path.join(self.cbf_dir, f"cbf_map.nii.gz"))
        except Exception as e:
            self.loggers.errors(f"Problem copying cbf_dir input {e}")

    def prepare_labels_inputs(self):
        """
        Prepare ROI label .txt file input
        """
        # Inputs provided
        if not self.parameters["run_mesh_mapping"]:
             # Prepare input image
            labels_file = self.check_input_file("labels_file", ext=".txt")
            self.labels_file = os.path.join(self.input_dir, "labels.txt")
            shutil.copy(labels_file, self.labels_file)
        # Inputs from prior processing step
        else:
            self.labels_file = os.path.join(self.output_dir, "labels.txt")

    def prepare_bc_inputs(self):
        """
        Prepare ROI label .txt file input
        """
        # Inputs provided
        if self.parameters["bc_file"]:
            # Prepare input image
            bc_file = self.check_input_file("bc_file", ext=".csv")
            self.bc_file = os.path.join(self.input_dir, "boundary_conditions.csv")
            shutil.copy(bc_file, self.bc_file)
        # Standard inputs
        else:
            self.bc_file = "/app/assets/boundary_conditions.csv"

    def prepare_inputs(self):
        """
        Begin input preparation
        """
        self.loggers.plugin_log("Preparing inputs")

        # Image inputs
        if any([
            self.parameters["run_preprocessing"],
            self.parameters["run_registration"],
            self.parameters["run_cortical_segmentation"],
        ]):
            self.prepare_input_image()


        # Global surface generation
        if self.parameters["run_surface_generation"]:
            self.prepare_seg_inputs()

        # Mesh generation
        # if self.parameters["run_mesh_generation"]:
        #     self.prepare_surface_inputs()
            
        # Mesh mapping
        if self.parameters["run_mesh_mapping"]:
            # Mesh inputs
            self.prepare_mesh_inputs()
            # DWI inputs
            if self.parameters["adjust_labels_dwi"] or self.parameters["generate_fa_map"]:
                self.prepare_dwi_inputs()
            # CBF inputs
            if self.parameters["generate_cbf_map"]:
                self.prepare_cbf_inputs()

        # Modelling inputs
        if self.parameters["run_modelling"]:
            if not self.parameters["run_mesh_mapping"]:
                self.prepare_mesh_inputs()
            self.prepare_surface_inputs(fixed_surfaces=True)
            self.prepare_labels_inputs()
            self.prepare_bc_inputs()