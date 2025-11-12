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

    def prepare_input_image(self):
        """
        Copy input image and atlas to working directory
        """
        # No input_im provided
        if not self.parameters["input_im"]:
            if self.parameters["run_preprocessing"] or self.parameters["run_registration"]:
                self.loggers.errors(f"An input image must be provided with --input_im")
            else:
                if not self.parameters["freesurfer_outputs"]:
                    self.loggers.errors(f"Either an input image or FreeSurfer output directory "
                                        f"must be provided with --input_im/freesurfer_outputs")
                else:
                    # FreeSurfer outputs provided instead
                    if not os.path.isdir(self.parameters["freesurfer_outputs"]):
                        self.loggers.errors(f"FreeSurfer outputs directory --freesurfer_outputs "
                                            f"does not exist {self.parameters['freesurfer_outputs']}")
                    else:
                        self.fs_outputs = os.path.join(self.input_dir, "fs_outputs")
                        shutil.copytree(self.parameters["freesurfer_outputs"], self.fs_outputs)
        else:
            # Not a valid directory
            if not os.path.isfile(self.parameters["input_im"]):
                self.loggers.errors(f"Input image provided with --input_im does "
                                    f"not exist {self.parameters['input_im']}")
            else:
                # Copy to working directory
                shutil.copy(self.parameters["input_im"], os.path.join(self.input_dir, "image.nii.gz"))

        # Atlas image checking
        if not os.path.isfile(self.parameters["input_atlas"]):
            self.loggers.errors(f"Input atlas provided with --input_atlas does "
                                f"not exist {self.parameters['input_atlas']}")
        else:
            shutil.copy(self.parameters["input_atlas"], os.path.join(self.input_dir, "atlas.nii.gz"))
            
            
    def prepare_seg_inputs(self):
        """
        Copy segmentation inputs to working directory
        """
        self.segmentation_dir = os.path.join(self.input_dir, "segmentations")
        os.makedirs(self.segmentation_dir)
        
        # No segmentation_dir provided and segmentations not due to run
        if not self.parameters["run_cortical_segmentation"]:
            if not self.parameters["segmentation_dir"] and not self.parameters["segmentations"]:
                self.loggers.errors(f"A segmentation directory --segmentation_dir or comma seperated list of segmentation "
                                    f"files --segmentations must be provded if --run_cortical_segmentation is false")
            else:
                if self.parameters["segmentation_dir"]:
                    # Not a valid directory
                    if not os.path.isdir(self.parameters["segmentation_dir"]):
                        self.loggers.errors(f"Segmentation input directory --segmentation_dir does not exist {self.parameters['segmentation_dir']}")
                    else:
                        # Copy to working directory
                        shutil.copytree(self.parameters["segmentation_dir"], self.segmentation_dir)
                elif self.parameters["segmentations"]:
                    segmentations = self.parameters["segmentations"].split(",")
                    for seg in segmentations:
                        if not os.path.isfile(seg):
                            self.loggers.errors(f"Segmentation path provided does not exist {seg}")
                        else:
                            # Copy to working directory
                            shutil.copy(seg, os.path.join(self.segmentation_dir, os.path.basename(seg)))
                
    def prepare_mesh_inputs(self):
        """
        Copy mesh inputs to working directory
        """
        self.mesh_dir = os.path.join(self.input_dir, "meshes")
        
        # No mesh_dir provided
        if not self.parameters["mesh_dir"]:
            self.loggers.errors(f"Global and regional mesh .vtk files must be provided with --mesh_dir")
        else:
            # Not a valid directory
            if not os.path.isdir(self.parameters["mesh_dir"]):
                self.loggers.errors(f"Mesh input directory --mesh_dir does not exist {self.parameters['mesh_dir']}")
            else:
                # Copy to working directory
                if self.parameters["run_mesh_mapping"]:
                    shutil.copytree(self.parameters["mesh_dir"], self.mesh_dir)
                else:
                    shutil.copytree(self.parameters["mesh_dir"], os.path.join(self.mesh_dir, "global"))

    def prepare_surface_inputs(self):
        """
        Copy surface inputs to working directory
        """
        self.surface_dir = os.path.join(self.input_dir, "surface_files")
        
        if self.parameters["adjust_outer_labels"]:
            if not self.parameters["outer_surface_fpath"]:
                self.loggers.errors(f"An outer surface (wholebrain) .stl file must be provided "
                                    f"with --outer_surface_fpath if --adjust_outer_labels is true")
            else:
                if not os.path.isfile(self.parameters["outer_surface_fpath"]):
                    self.loggers.errors(f"Outer surface .stl file --outer_surface_fpath does not exist {self.parameters['outer_surface_fpath']}")
                else:
                    # Copy to working directory
                    shutil.copy(self.parameters["outer_surface_fpath"], self.surface_dir)
        
    def prepare_dwi_inputs(self):
        """
        Prepare diffusion weighted imaging inputs
        """
        self.dwi_dir = os.path.join(self.input_dir, "dwi_files")
        os.makedirs(self.dwi_dir, exist_ok=True)
        
        if self.parameters["adjust_labels_dwi"] or self.parameters["generate_fa_map"]:
            if not self.parameters["dwi_dir"]:
                self.loggers.errors(f"Directory containing diffusion weighted imaging files must be provided "
                                    f"with --dwi_dir if --adjust_labels_dwi or --generate_fa_map is True")
            else:
                if not os.path.isdir(self.parameters["dwi_dir"]):
                    self.loggers.errors(f"DWI input directory --dwi_dir does not exist {self.parameters['dwi_dir']}")
                else:
                    # Copy to working directory
                    dwi_dir = self.parameters["dwi_dir"]
                    shutil.copy(glob.glob(os.path.join(dwi_dir, "*tensor*.nii*"))[0], os.path.join(self.dwi_dir, f"dwi_tensor.nii.gz"))
                    shutil.copy(glob.glob(os.path.join(dwi_dir, "*L1*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_L1.nii.gz"))
                    shutil.copy(glob.glob(os.path.join(dwi_dir, "*L2*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_L2.nii.gz"))
                    shutil.copy(glob.glob(os.path.join(dwi_dir, "*L3*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_L3.nii.gz"))
                    shutil.copy(glob.glob(os.path.join(dwi_dir, "*FA*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_FA.nii.gz"))
                    shutil.copy(glob.glob(os.path.join(dwi_dir, "*MD*.nii.gz"))[0], os.path.join(self.dwi_dir, f"dwi_MD.nii.gz"))
                
    def prepare_cbf_inputs(self):
        """
        Prepare cerebral blood flow imaging inputs
        """
        self.cbf_dir = os.path.join(self.input_dir, "cbf_files")
        os.makedirs(self.cbf_dir, exist_ok=True)
        
        if self.parameters["generate_cbf_map"]:
            if not self.parameters["cbf_dir"]:
                self.loggers.errors(f"Directory containing cerebral blood flow imaging files must be provided "
                                    f"with --cbf_dir if --generate_cbf_map is True")
            else:
                if not os.path.isdir(self.parameters["cbf_dir"]):
                    self.loggers.errors(f"CBF input directory --cbf_dir does not exist {self.parameters['cbf_dir']}")
                else:
                    # Copy to working directory
                    cbf_dir = self.parameters["cbf_dir"]
                    shutil.copy(glob.glob(os.path.join(cbf_dir, "*.nii.gz"))[0], os.path.join(self.cbf_dir, f"cbf_map.nii.gz"))

    def prepare_labels_inputs(self):
        """
        Prepare ROI label .txt file input
        """
        if not self.parameters["labels_fpath"]:
            self.loggers.errors("If --run_mesh_mapping set to False, ROI label .txt file must be provided with --labels_fpath")
        elif not os.path.isfile(self.parameters["labels_fpath"]):
            self.loggers.errors(f"No valid ROI label .txt file found at {self.parameters['labels_fpath']}")
        else: 
            shutil.copy(self.parameters["labels_fpath"], os.path.join(self.input_dir, "labels.txt"))
                
    def prepare_bc_inputs(self):
        """
        Prepare boundary condition file input
        """
        if self.parameters["bc_fpath"]:
            if not os.path.isfile(self.parameters["bc_fpath"]):
                self.loggers.errors(f"No valid boundary condition .csv file found at {self.parameters['bc_fpath']}")
            else:
                self.boundary_conditions = os.path.join(self.input_dir, f"boundary_conditions.csv")
                shutil.copy(self.parameters["bc_fpath"], self.boundary_conditions)
        else:
            self.boundary_conditions = "/app/assets/boundary_conditions.csv"

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
            
        # Cortical segmentation inputs
        if self.parameters["run_mesh_mapping"]:
            self.prepare_mesh_inputs()
            self.prepare_surface_inputs()
            self.prepare_dwi_inputs()
            self.prepare_cbf_inputs()

        # Modelling inputs
        if self.parameters["run_modelling"]:
            if not self.parameters["run_mesh_mapping"]:
                self.prepare_labels_inputs()
                self.prepare_mesh_inputs()
            self.prepare_bc_inputs()