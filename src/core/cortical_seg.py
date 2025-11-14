# Imports
import os
import sys
import subprocess
import nibabel as nib
import numpy as np
import ants
import shutil

class FreeSurfer(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["loggers", "parameters", "base_dir", "freesurfer_env",
                      "input_dir", "interim_dir", "output_dir", "log_dir", "tmp_dir"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def build_freesurfer_command(self):
        """
        Build freesurfer command.
        """
        # Define command
        self.subject_id = self.parameters["subject_id"]
        self.freesurfer_source  = "source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        self.freesurfer_command = f"recon-all -i {self.input_im} -subject {self.subject_id} " + \
                                  f"-all -parallel -norandomness"

        # Add extra flags
        for _input, tag in {"big_vents"          : "bigventricles",
                            "large_FOV"          : "cw256"}.items():

            if self.parameters[_input]:
                self.freesurfer_command += f" -{tag}"

    def run_freesurfer(self):
        """
        Run Freesurfer.
        """
        self.loggers.plugin_log("Starting execution")
        self.freesurfer_log     = os.path.join(self.log_dir, "freesurfer.log")
        self.freesurfer_command = self.freesurfer_source + self.freesurfer_command
        self.loggers.plugin_log(f"Freesurfer command: {self.freesurfer_command}")
        with open(self.freesurfer_log, "w") as outfile:
            freesurfer_sub = subprocess.run(["bash", "-c",
                                             self.freesurfer_command],
                                             stdout=outfile,
                                             stderr=subprocess.STDOUT,
                                             env=self.freesurfer_env)

        # Copy outputs regardless of success
        shutil.copytree(os.path.join(self.tmp_dir, self.parameters["subject_id"]), 
                        os.path.join(self.interim_dir, "fs_outputs"))
        if os.path.isdir(os.path.join(self.interim_dir, "fs_outputs")):
            shutil.rmtree(os.path.join(self.tmp_dir, self.subject_id))

        # Produce error if failed to process
        if freesurfer_sub.returncode != 0:
            self.loggers.errors(f"Freesurfer execution returned non-zero exit status - " +
                                f"please check log file at {self.freesurfer_log}")

        # Check required outputs have been produced
        segmentation = os.path.join(os.path.join(self.interim_dir, "fs_outputs"), "mri", "aseg.mgz")
        output_im    = os.path.join(os.path.join(self.interim_dir, "fs_outputs"), "mri", "T1.mgz")
        if not os.path.exists(segmentation):
            self.loggers.errors(f"Freesurfer has not produced a segmentation at {segmentation}" +
                                f"- please check log file at {self.freesurfer_log}")
        elif not os.path.exists(output_im):
            self.loggers.errors(f"Freesurfer has not produced an output image at {output_im} - " +
                                f"please check log file at {self.freesurfer_log}")
        else:
            self.loggers.plugin_log("Freesurfer run successfully")

    def convert_T1(self):
        """
        Convert freesurfer T1.mgz to .nii.gz
        """
        # Define image and log paths
        T1 = os.path.join(self.fs_outputs, "mri", "T1.mgz")
        self.T1_out  = os.path.join(self.fs_outputs, "mri", "T1.nii.gz")
        conversion_log = os.path.join(self.log_dir, "T1_conversion.log")

        # Convert
        with open(conversion_log, "w") as outfile:
            subprocess.run(["bash", "-c",
                            self.freesurfer_source + "mri_convert " +
                            f"{T1} {self.T1_out}"],
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            env=self.freesurfer_env)
            
        # Check if conversion successful
        if not os.path.exists(self.T1_out):
            self.loggers.errors("Conversion of T1.mgz to T1.nii.gz failed - " +
                               f"please check log file at {conversion_log}")
        else:
            shutil.copy(self.T1_out, os.path.join(self.output_dir, "fs_im.nii.gz"))

    def convert_seg(self):
        """
        Convert freesurfer aseg.mgz to .nii.gz
        """
        # Define image and log paths
        seg = os.path.join(self.fs_outputs, "mri", "aseg.mgz")
        self.seg_out  = os.path.join(self.fs_outputs, "mri", "aseg.nii.gz")
        conversion_log = os.path.join(self.log_dir, "aseg_conversion.log")

        # Convert
        with open(conversion_log, "w") as outfile:
            subprocess.run(["bash", "-c",
                            self.freesurfer_source + "mri_convert " +
                            f"{seg} {self.seg_out}"],
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            env=self.freesurfer_env)
            
        # Check if conversion successful
        if not os.path.exists(self.seg_out):
            self.loggers.errors("Conversion of aseg.mgz to aseg.nii.gz failed - " +
                               f"please check log file at {conversion_log}")

    def binarise(self, region):
        """
        Binarise to extract required freesurfer labels
        """
        # Define image and log paths
        subcortical_seg = os.path.join(self.fs_outputs, "mri", "aseg.nii.gz")
        bin_out  = os.path.join(self.interim_dir, f"{region}", f"{region}_bin.nii.gz")
        os.makedirs(os.path.join(self.interim_dir, region), exist_ok=True)
        binarise_log = os.path.join(self.log_dir, f"binarise.log")

        labels = {"ventricles"     : "24 4 5 14 15 43 44 213",
                  "cerebellum_L"   : "6 7 8",
                  "cerebellum_R"   : "45 46 47",
                  "cerebellumWM_L" : "7",
                  "cerebellumWM_R" : "46",
                  "brainstem"      : "16 170 171 172 173 174 175 177 178 179 71000 71010",
                  "cerebrum_L"     : "2 3 10 11 12 13 17 18 19 20 26 28",
                  "cerebrum_R"     : "41 42 49 50 51 52 53 54 55 56 58 60",
                  "cerebrumWM_L"   : "2 78",
                  "cerebrumWM_R"   : "41 79",
                  "wholebrain"     : "6 7 8 16 45 46 47 192 "
                                     "24 4 5 14 15 43 44 213 "
                                     "2 3 10 11 12 13 17 18 19 20 26 28 "
                                     "41 42 49 50 51 52 53 54 55 56 58 60"
                 }

        # Convert
        with open(binarise_log, "w") as outfile:
            subprocess.run(["bash", "-c",
                            self.freesurfer_source + "mri_binarize " +
                            f"--i {subcortical_seg} --match {labels[region]}" +
                            f"--inv --o {bin_out}"],
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            env=self.freesurfer_env)
            
        # Check if conversion successful
        if not os.path.exists(bin_out):
            self.loggers.errors(f"Binarisation of {region} segmentation failed - " +
                                f"please check log file at {binarise_log}")

    def register_mni_atlas(self):
        """
        Register MNI-ICBM152 CerebrA atlas labels to subject space.
    
        Reference:
            Manera AL, Dadar M, Fonov V, Collins DL. (2020).
            CerebrA, registration and manual label correction of Mindboggle-101 atlas
            for MNI-ICBM152 template. Scientific Data, 7, 237.
            https://doi.org/10.1038/s41597-020-00564-0
        """
        # Set paths
        atlas_t1 = "/app/assets/mni_icbm152_atlas_t1.nii.gz"
        atlas_labels = "/app/assets/mni_icbm152_CerebrA_atlas_labels.nii.gz"
        atlas_labels_out = os.path.join(self.interim_dir, "mni_icbm152_labels_subjectspace.nii.gz")
        brainstem_seg = os.path.join(self.interim_dir, "brainstem", "brainstem_bin.nii.gz")
        input_image = self.input_im if self.input_im else os.path.join(self.fs_outputs, "mri", "T1.nii.gz")
        
        # Register atlas T1 to subject T1 space
        registration = ants.registration(fixed=ants.image_read(input_image), 
                                         moving=ants.image_read(atlas_t1), 
                                         type_of_transform="Affine")

        # Apply transform to atlas labels
        transformed_labels = ants.apply_transforms(fixed=ants.image_read(input_image),
                                                   moving=ants.image_read(atlas_labels),
                                                   transformlist=[registration["fwdtransforms"][0]],
                                                   interpolator="nearestNeighbor")

        # Resample labels so dimensions match brainstem seg
        resampled = ants.resample_image_to_target(image=transformed_labels,
                                                  target=ants.image_read(brainstem_seg),
                                                  interp_type="nearestNeighbor")
                
        # Save transformed labels
        ants.image_write(resampled, atlas_labels_out)
        self.atlas_in_subj = atlas_labels_out

        # Check outputs
        if not os.path.exists(atlas_labels_out):
            self.loggers.errors("Transformation of atlas labels to subject space failed")

    def split_brainstem(self):
        """
        Split the binarised brainstem into left and right hemispheres
        """        
        # Load brainstem segmentation
        brainstem_seg = nib.load(os.path.join(self.interim_dir, "brainstem", "brainstem_bin.nii.gz"))
        brainstem_data = brainstem_seg.get_fdata().astype(np.uint8)
        affine = brainstem_seg.affine
        
        # Load atlas labels
        atlas_img = nib.load(self.atlas_in_subj)
        atlas_data = atlas_img.get_fdata().astype(int)
        
        # Create masks
        mask = brainstem_data > 0
        left_mask = np.zeros_like(brainstem_data, dtype=np.uint8)
        right_mask = np.zeros_like(brainstem_data, dtype=np.uint8)
        
        # Assign voxels that overlap atlas left/right IDs
        left_vox = np.isin(atlas_data, [62]) & mask
        right_vox = np.isin(atlas_data, [11]) & mask
        left_mask[left_vox] = 1
        right_mask[right_vox] = 1
        
        # Find leftover voxels (in aseg but not in atlas L/R)
        assigned = left_vox | right_vox
        leftovers = mask & (~assigned)
        
        if np.any(leftovers):
            coords = np.array(np.nonzero(leftovers)).T
            ras_coords = nib.affines.apply_affine(affine, coords)
        
            for (i, j, k), ras in zip(coords, ras_coords):
                if ras[0] < 0:   # Left side
                    left_mask[i, j, k] = 1
                else:            # Right side
                    right_mask[i, j, k] = 1
        
        # Save outputs
        left_output = os.path.join(self.interim_dir, "brainstem_L", "brainstem_L_bin.nii.gz")
        right_output = os.path.join(self.interim_dir, "brainstem_R", "brainstem_R_bin.nii.gz")
        for file in [left_output, right_output]:
            _dir = os.path.dirname(file)
            os.makedirs(_dir, exist_ok=True)
            
        nib.save(nib.Nifti1Image(left_mask, affine, brainstem_seg.header), left_output)
        nib.save(nib.Nifti1Image(right_mask, affine, brainstem_seg.header), right_output)

        # Check outputs
        if not os.path.exists(left_output):
            self.loggers.errors(f"Splitting of brainstem region failed")
        elif not os.path.exists(right_output):
            self.loggers.errors(f"Splitting of brainstem region failed")

    def run_cortical_seg(self):
        """
        Run FreeSurfer cortical segmentation
        """
        # Define parameters
        self.subject_id = self.parameters["subject_id"]
        self.interim_dir = os.path.join(self.interim_dir, "segmentation")
        os.makedirs(self.interim_dir, exist_ok=True)
        regions = self.parameters["regions"].split(",")

        # Define inputs
        if not self.parameters["freesurfer_outputs"]:
            self.fs_outputs = os.path.join(self.interim_dir, "fs_outputs")
            if self.parameters["run_registration"]:
                self.input_im = os.path.join(self.output_dir, "registered_im.nii.gz")
            elif self.parameters["run_preprocessing"]:
                self.input_im = os.path.join(self.output_dir, "image.nii.gz")
            else:
                self.input_im = os.path.join(self.input_dir, "image.nii.gz")

            # Run Freesurfer
            self.loggers.plugin_log("Running Freesurfer")
            self.build_freesurfer_command()
            self.run_freesurfer()

        # FreeSurfer outputs already provided    
        else:
            self.fs_outputs = os.path.join(self.input_dir, "fs_outputs")

        # Convert aseg file to NIfTI
        self.loggers.plugin_log("Converting aseg and T1 file to NIfTI")
        self.convert_T1()
        self.convert_seg()

        # Create region binary files
        self.loggers.plugin_log("Creating region binary files")
        # Join brainstem label - split in later steps
        if "brainstem_L" in regions or "brainstem_R" in regions:
            binarise_regions = [r for r in regions if r not in ("brainstem_L", "brainstem_R")]
            binarise_regions.append("brainstem")
        # Process regions
        for region in binarise_regions:
            self.binarise(region)

        # Register MNI-ICBM152 atlas labels to subject space
        self.loggers.plugin_log("Registering atlas labels to subject space")
        self.register_mni_atlas()

        # Split the brainstem into L&R
        self.loggers.plugin_log("Splitting brainstem into L&R")
        self.split_brainstem()

        # Copy to output directory
        outpath = os.path.join(self.output_dir, "segmentations")
        os.makedirs(outpath, exist_ok=True)
        for region in regions:
            shutil.copy(os.path.join(self.interim_dir, region, f"{region}_bin.nii.gz"), outpath)