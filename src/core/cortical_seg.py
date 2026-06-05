# Imports
import os
import sys
import shutil
import subprocess
import nibabel as nib
import numpy as np
import ants


class CorticalSeg(object):
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
            "regions",
            "region_definitions",
            "synthseg_env",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def resolve_segmentation_regions(self):
        """
        Resolve the segmentation masks required for the requested regions.

        Returns:
        ---
        segmentation_regions (list)   : Segmentation regions that must be binarised.
        split_brainstem (bool)        : True if lateralised brainstem masks are required.
        """
        segmentation_regions = []
        split_brainstem = False

        for region_name in self.regions:
            # Get region definition
            region_definition = self.region_definitions[region_name]

            # If region is a segmentation, add to list of regions to binarise
            if region_definition["region_type"] == "segmentation":
                if region_name not in segmentation_regions:
                    segmentation_regions.append(region_name)
                continue

            # If region is brainstem_L or brainstem_R, set flag to split brainstem
            if region_name in ("brainstem_L", "brainstem_R"):
                split_brainstem = True
                if "brainstem" not in segmentation_regions:
                    segmentation_regions.append("brainstem")
                continue

            # If region makes up a derived region, add source regions to list of regions to binarise
            for source_region in region_definition.get("combine_regions", []) + region_definition.get("subtract_regions", []):
                source_definition = self.region_definitions.get(source_region)
                if source_definition is None:
                    self.loggers.errors(f"Unknown source region {source_region} for derived region {region_name}")
                if source_definition["region_type"] != "segmentation":
                    self.loggers.errors(
                        f"Derived region {region_name} depends on non-segmentation source region {source_region}"
                    )
                if source_region not in segmentation_regions:
                    segmentation_regions.append(source_region)

        return segmentation_regions, split_brainstem

    def build_synthseg_command(self):
        """
        Build SynthSeg command.
        """
        # Define paths and parameters
        self.synthseg_source = "source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        self.synthseg_outdir = os.path.join(self.interim_dir, "synthseg_outputs")
        os.makedirs(self.synthseg_outdir, exist_ok=True)
        self.synthseg_output = os.path.join(self.synthseg_outdir, "synthseg_output.nii.gz")

        # Build command
        self.synthseg_command = f"mri_synthseg --i {self.input_im} --o {self.synthseg_output}"
        if not self.parameters["use_gpu"]:
            self.synthseg_command += " --cpu"

    def run_synthseg(self):
        """
        Run SynthSeg.
        """
        # Define command and log path
        self.synthseg_log = os.path.join(self.log_dir, "segmentation.log")
        command = self.synthseg_source + self.synthseg_command
        self.loggers.verbose_log(f"SynthSeg command: {command}")

        # Run SynthSeg
        with open(self.synthseg_log, "a") as outfile:
            synthseg_sub = subprocess.run(
                ["bash", "-c", command],
                stdout=outfile,
                stderr=subprocess.STDOUT,
                env=self.synthseg_env,
            )

        # Check if execution was successful
        if synthseg_sub.returncode != 0:
            self.loggers.errors(
                f"SynthSeg execution returned non-zero exit status - "
                f"please check log file at {self.synthseg_log}"
            )

        # Check if output file was created
        if not os.path.exists(self.synthseg_output):
            self.loggers.errors(
                f"SynthSeg has not produced a segmentation at {self.synthseg_output} - "
                f"please check log file at {self.synthseg_log}"
            )
        else:
            self.loggers.verbose_log("SynthSeg execution successful")

    def build_nextbrain_command(self):
        """
        Build NextBrain command.
        """
        # Define paths and parameters
        self.nextbrain_source = "source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        self.nextbrain_outdir = os.path.join(self.interim_dir, "nextbrain_outputs")
        os.makedirs(self.nextbrain_outdir, exist_ok=True)

        # FreeSurfer 8.1.0 positional CLI runs both sides in one command.
        gpu_flag = 1 if self.parameters["use_gpu"] else 0
        self.nextbrain_command = (
            f"mri_histo_atlas_segment_fireants {self.input_im} {self.nextbrain_outdir} "
            f"{gpu_flag} -1"
        )

    def run_nextbrain(self):
        """
        Run NextBrain.
        """
        # Define command and log path
        self.nextbrain_log = os.path.join(self.log_dir, "segmentation.log")
        command = self.nextbrain_source + self.nextbrain_command
        self.loggers.verbose_log(f"NextBrain command: {command}")

        # Run NextBrain
        with open(self.nextbrain_log, "a") as outfile:
            nextbrain_sub = subprocess.run(
                ["bash", "-c", command],
                stdout=outfile,
                stderr=subprocess.STDOUT,
                env=self.synthseg_env,
            )

        # Check if execution was successful
        if nextbrain_sub.returncode != 0:
            self.loggers.errors(
                f"NextBrain execution returned non-zero exit status - "
                f"please check log file at {self.nextbrain_log}"
            )

        # Assign side-specific label images for direct L/R NextBrain extraction.
        self.nextbrain_side_outputs = {
            "L": os.path.join(self.nextbrain_outdir, "seg.left.nii.gz"),
            "R": os.path.join(self.nextbrain_outdir, "seg.right.nii.gz"),
        }
        for side, side_output in self.nextbrain_side_outputs.items():
            if not os.path.exists(side_output):
                self.loggers.errors(
                    f"NextBrain has not produced the {side} segmentation at {side_output} - "
                    f"please check log file at {self.nextbrain_log}"
                )

        self.loggers.verbose_log("NextBrain execution successful")

    def binarise(self, region: str, segmentation_output: str, label_key: str, segmentation_name: str):
        """
        Create a binary mask for a segmentation region from an atlas segmentation output.

        Parameters:
        ---
        region (str)                : Name of the segmentation region to extract.
        segmentation_output (str)   : Path to the source atlas segmentation.
        label_key (str)             : Region definition key containing atlas labels.
        segmentation_name (str)     : Name of the atlas segmentation source.
        """
        # Define paths
        bin_out = os.path.join(self.interim_dir, region, f"{region}_bin.nii.gz")
        os.makedirs(os.path.join(self.interim_dir, region), exist_ok=True)
        binarise_log = os.path.join(self.log_dir, "segmentation.log")

        labels = self.region_definitions.get(region, {}).get(label_key)
        if not labels:
            self.loggers.errors(f"No {segmentation_name} labels defined for segmentation region {region}")

        label_string = " ".join(str(label) for label in labels)
        freesurfer_source = "source $FREESURFER_HOME/SetUpFreeSurfer.sh && "

        # Binarise using mri_binarize
        with open(binarise_log, "a") as outfile:
            binarise_sub = subprocess.run(
                [
                    "bash",
                    "-c",
                    freesurfer_source
                    + "mri_binarize "
                    + f"--i {segmentation_output} --match {label_string} --o {bin_out}",
                ],
                stdout=outfile,
                stderr=subprocess.STDOUT,
                env=self.synthseg_env,
            )

        # Check if execution was successful
        if binarise_sub.returncode != 0:
            self.loggers.errors(
                f"Binarisation of {region} {segmentation_name} segmentation returned non-zero exit status - "
                f"please check log file at {binarise_log}"
            )

        # Check if output file was created
        if not os.path.exists(bin_out):
            self.loggers.errors(
                f"Binarisation of {region} {segmentation_name} segmentation failed - "
                f"please check log file at {binarise_log}"
            )

    def register_mni_atlas(self):
        """
        Register MNI-ICBM152 CerebrA atlas labels to subject space.

        Reference:
            Manera AL, Dadar M, Fonov V, Collins DL. (2020)
            CerebrA, registration and manual label correction of Mindboggle-101 atlas
            for MNI-ICBM152 template. Scientific Data, 7, 237
            https://doi.org/10.1038/s41597-020-00564-0
        """
        # Define paths
        atlas_t1 = "/app/assets/mni_icbm152_atlas_t1.nii.gz"
        atlas_labels = "/app/assets/mni_icbm152_CerebrA_atlas_labels.nii.gz"
        atlas_labels_out = os.path.join(self.interim_dir, "mni_icbm152_labels_subjectspace.nii.gz")
        brainstem_seg = os.path.join(self.interim_dir, "brainstem", "brainstem_bin.nii.gz")

        # Register atlas T1 to subject space
        registration = ants.registration(
            fixed=ants.image_read(self.input_im),
            moving=ants.image_read(atlas_t1),
            type_of_transform="Affine",
        )

        # Apply the transform to atlas labels
        transformed_labels = ants.apply_transforms(
            fixed=ants.image_read(self.input_im),
            moving=ants.image_read(atlas_labels),
            transformlist=[registration["fwdtransforms"][0]],
            interpolator="nearestNeighbor",
        )

        # Resample to match brainstem segmentation space
        resampled = ants.resample_image_to_target(
            image=transformed_labels,
            target=ants.image_read(brainstem_seg),
            interp_type="nearestNeighbor",
        )

        # Save the transformed and resampled atlas labels
        ants.image_write(resampled, atlas_labels_out)
        self.atlas_in_subj = atlas_labels_out

        # Check output
        if not os.path.exists(atlas_labels_out):
            self.loggers.errors("Transformation of atlas labels to subject space failed")

    def split_brainstem(self):
        """
        Split the binarised brainstem into left and right masks.
        """
        # Load brainstem segmentation
        brainstem_seg = nib.load(os.path.join(self.interim_dir, "brainstem", "brainstem_bin.nii.gz"))
        brainstem_data = brainstem_seg.get_fdata().astype(np.uint8)
        affine = brainstem_seg.affine

        # Load atlas labels
        atlas_img = nib.load(self.atlas_in_subj)
        atlas_data = atlas_img.get_fdata().astype(int)

        # Create left and right masks
        mask = brainstem_data > 0
        left_mask = np.zeros_like(brainstem_data, dtype=np.uint8)
        right_mask = np.zeros_like(brainstem_data, dtype=np.uint8)
        left_vox = np.isin(atlas_data, [62]) & mask
        right_vox = np.isin(atlas_data, [11]) & mask
        left_mask[left_vox] = 1
        right_mask[right_vox] = 1

        # Identify and remaining unassigned voxels
        assigned = left_vox | right_vox
        leftovers = mask & (~assigned)

        # Assign based on RAS coordinates
        if np.any(leftovers):
            coords = np.array(np.nonzero(leftovers)).T
            ras_coords = nib.affines.apply_affine(affine, coords)

            for (i, j, k), ras in zip(coords, ras_coords):
                if ras[0] < 0:
                    left_mask[i, j, k] = 1
                else:
                    right_mask[i, j, k] = 1

        # Save left and right brainstem masks
        left_output = os.path.join(self.interim_dir, "brainstem_L", "brainstem_L_bin.nii.gz")
        right_output = os.path.join(self.interim_dir, "brainstem_R", "brainstem_R_bin.nii.gz")
        for file in [left_output, right_output]:
            _dir = os.path.dirname(file)
            os.makedirs(_dir, exist_ok=True)

        nib.save(nib.Nifti1Image(left_mask, affine, brainstem_seg.header), left_output)
        nib.save(nib.Nifti1Image(right_mask, affine, brainstem_seg.header), right_output)

        # Check outputs
        if not os.path.exists(left_output):
            self.loggers.errors("Splitting of brainstem region failed")
        elif not os.path.exists(right_output):
            self.loggers.errors("Splitting of brainstem region failed")

    def create_global_mask(self):
        """
        Create a global binary mask from wholebrain minus ventricles.
        """
        # Define paths
        wholebrain_path = os.path.join(self.interim_dir, "wholebrain", "wholebrain_bin.nii.gz")
        ventricles_path = os.path.join(self.interim_dir, "ventricles", "ventricles_bin.nii.gz")
        global_dir = os.path.join(self.interim_dir, "global")
        global_path = os.path.join(global_dir, "global.nii.gz")
        os.makedirs(global_dir, exist_ok=True)

        # Load wholebrain and ventricles segmentations, create global mask
        wholebrain_img = nib.load(wholebrain_path)
        wholebrain_data = wholebrain_img.get_fdata() > 0
        ventricles_data = nib.load(ventricles_path).get_fdata() > 0
        global_data = np.logical_and(wholebrain_data, np.logical_not(ventricles_data)).astype(np.uint8)

        # Save global mask
        nib.save(
            nib.Nifti1Image(global_data, wholebrain_img.affine, wholebrain_img.header),
            global_path,
        )

        if not os.path.exists(global_path):
            self.loggers.errors("Creation of global segmentation mask failed")

    def run_cortical_seg(self):
        """
        Run cortical segmentation and generate required binary region masks.
        """
        self.loggers.plugin_log("Running cortical segmentation")
        # Create interim directory for segmentation outputs
        self.interim_dir = os.path.join(self.interim_dir, "segmentation")
        os.makedirs(self.interim_dir, exist_ok=True)

        # Set up paths and parameters for atlas segmentations
        segmentation_regions, split_brainstem = self.resolve_segmentation_regions()
        for required_region in ["wholebrain", "ventricles"]:
            if required_region not in segmentation_regions:
                segmentation_regions.append(required_region)

        synthseg_regions = []
        nextbrain_regions = []
        for region in segmentation_regions:
            region_definition = self.region_definitions.get(region, {})
            has_synthseg_labels = bool(region_definition.get("synthseg_labels"))
            has_nextbrain_labels = bool(region_definition.get("nextbrain_labels"))

            if has_synthseg_labels:
                synthseg_regions.append(region)
            elif has_nextbrain_labels:
                nextbrain_regions.append(region)
            else:
                self.loggers.errors(
                    f"Segmentation region {region} must define synthseg_labels or nextbrain_labels"
                )

        # Determine input image for atlas segmentation based on which steps have been run
        if self.parameters["run_registration"] or self.parameters["run_preprocessing"]:
            self.input_im = os.path.join(self.output_dir, "image.nii.gz")
        else:
            self.input_im = os.path.join(self.input_dir, "image.nii.gz")
        self.loggers.verbose_log(f"Segmentation input image: {self.input_im}")

        # Run required atlas segmentations
        if synthseg_regions:
            self.loggers.verbose_log("Running SynthSeg")
            self.build_synthseg_command()
            self.run_synthseg()

        if nextbrain_regions:
            self.loggers.verbose_log("Running NextBrain")
            self.build_nextbrain_command()
            self.run_nextbrain()

        # Binarise to extract required regions, and split brainstem if required
        self.loggers.verbose_log("Creating region binary files")
        for region in synthseg_regions:
            self.binarise(region, self.synthseg_output, "synthseg_labels", "SynthSeg")
        for region in nextbrain_regions:
            side = self.region_definitions[region].get("side")
            if side not in self.nextbrain_side_outputs:
                self.loggers.errors(f"NextBrain region {region} must resolve to side L or R")
            nextbrain_output = self.nextbrain_side_outputs[side]
            self.binarise(region, nextbrain_output, "nextbrain_labels", "NextBrain")

        # If lateralised brainstem regions are required, split segmentation
        if split_brainstem:
            self.loggers.verbose_log("Registering atlas labels to subject space")
            self.register_mni_atlas()

            self.loggers.verbose_log("Splitting brainstem into L&R")
            self.split_brainstem()

        self.loggers.verbose_log("Creating global segmentation mask")
        self.create_global_mask()

        # Copy final segmentation outputs to output directory
        outpath = os.path.join(self.output_dir, "segmentations")
        os.makedirs(outpath, exist_ok=True)
        for region in segmentation_regions:
            shutil.copy(os.path.join(self.interim_dir, region, f"{region}_bin.nii.gz"), outpath)
        if split_brainstem:
            for region in ["brainstem_L", "brainstem_R"]:
                shutil.copy(os.path.join(self.interim_dir, region, f"{region}_bin.nii.gz"), outpath)
        shutil.copy(os.path.join(self.interim_dir, "global", "global.nii.gz"), outpath)
