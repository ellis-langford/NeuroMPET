# Imports
import os
import sys
import glob
import subprocess
import nibabel as nib
import numpy as np
import ants
import trimesh
import shutil

class SurfaceGen(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["loggers", "parameters", "base_dir", 
                      "input_dir", "interim_dir", "output_dir", "log_dir"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def tessellate(self, region):
        """
        Tessellate to create surface from input volume
        """
        # Define image and log paths
        bin_data = glob.glob(os.path.join(self.segmentation_dir, f"*{region}*.nii.gz"))[0]
        surf_out = os.path.join(self.interim_dir, f"{region}", "surf")
        tessellate_log = os.path.join(self.log_dir, f"tessellate_{region}.log")

        # Convert
        with open(tessellate_log, "w") as outfile:
            subprocess.run(["bash", "-c",
                            self.freesurfer_source + "mri_tessellate " +
                            f"{bin_data} 1 {surf_out}"],
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            env=self.freesurfer_env)
            
        # Check if conversion successful
        if not os.path.exists(surf_out):
            self.loggers.errors(f"Tessellation of {region} failed - " +
                                f"please check log file at {tessellate_log}")

    def smooth(self, region):
        """
        Smooths the tessellation of region surface
        """
        # Define image and log paths
        surfs = os.path.join(self.interim_dir, f"{region}", "surf")
        smooth_out = os.path.join(self.interim_dir, f"{region}", "smooth")
        smooth_log = os.path.join(self.log_dir, f"smooth_{region}.log")

        # Convert
        with open(smooth_log, "w") as outfile:
            subprocess.run(["bash", "-c",
                            self.freesurfer_source + "mris_smooth " +
                            f"{surfs} {smooth_out}"],
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            env=self.freesurfer_env)
            
        # Check if conversion successful
        if not os.path.exists(smooth_out):
            self.loggers.errors(f"Smoothing of {region} failed - " +
                                f"please check log file at {smooth_log}")
            
    def convert_to_stl(self, region):
        """
        Converts geometry file to .stl
        """
        input_im = os.path.join(self.interim_dir, f"{region}", "smooth")

        # Define output and log paths
        geo_out = os.path.join(self.interim_dir, f"{region}", f"{region}.stl")
        conversion_log = os.path.join(self.log_dir, f"{region}_conversion.log")

        # Convert
        with open(conversion_log, "w") as outfile:
            subprocess.run(["bash", "-c",
                            self.freesurfer_source + "mris_convert " +
                            f"{input_im} {geo_out}"],
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            env=self.freesurfer_env)
            
        # Check if conversion successful
        if not os.path.exists(geo_out):
            self.loggers.errors(f"Conversion {region} to .stl failed - " +
                                f"please check log file at {conversion_log}")

    def clean_stl(self, region):
        """
        Fix issues in stl file (e.g. non-normal orientations)
        """
        input_im = os.path.join(self.interim_dir, f"{region}", f"{region}.stl")

        # Define output and log paths
        geo_out = os.path.join(self.output_dir, "surfaces", f"{region}.stl")
        os.makedirs(os.path.join(self.output_dir, "surfaces"), exist_ok=True)

        # Clean
        mesh = trimesh.load(input_im) # Load mesh
        components = mesh.split(only_watertight=False) # Split into connected components
        largest = max(components, key=lambda c: c.area) # Keep largest shell
        
        # Remove small disconnected patches
        cleaned_components = [
            c for c in components if c.area > 0.05 * largest.area
        ]
        
        mesh_clean = trimesh.util.concatenate(cleaned_components)
        
        # Fill holes
        if not mesh_clean.is_watertight:
            mesh_clean.fill_holes()
        
        # Export cleaned STL
        mesh_clean.export(geo_out)
            
        # Check if conversion successful
        if not os.path.exists(geo_out):
            self.loggers.errors(f"Cleaning of {region} .stl failed")

    def prepare_geometry(self, region):
        """
        Binarises, tessellates, smooths, converts and cleans volume to .stl
        """
        # Outputdir
        os.makedirs(os.path.join(self.interim_dir, f"{region}"), exist_ok=True)
        
        # Process geometry
        self.tessellate(region)
        self.smooth(region)
        self.convert_to_stl(region)
        self.clean_stl(region)

        # Define expected outputs
        surf_out = os.path.join(self.interim_dir, f"{region}", "surf")
        smooth_out = os.path.join(self.interim_dir, f"{region}", "smooth")
        conv_out = os.path.join(self.interim_dir, f"{region}", f"{region}.stl")
        geo_out = os.path.join(self.output_dir, "surfaces", f"{region}.stl")
            
        # Check if geometry generation successful
        outputs = [surf_out, smooth_out, conv_out, geo_out]
        for output in outputs:
            if not os.path.exists(output):
                self.loggers.errors(f"Geometry generation failed - " +
                                    f"required output missing ({output})")

    def generate_global_surface(self):
        """
        Generate global mesh file
        (all regions minus ventricles)
        """
        if "wholebrain" not in self.regions or "ventricles" not in self.regions:
            self.loggers.errors("Wholebrain and ventricles segmentations must be provided if --generate_global is True")
            
        # Load global binary and ventricle mask
        wholebrain_seg = nib.load(glob.glob(os.path.join(self.segmentation_dir, "*wholebrain*.nii.gz"))[0])
        vent_seg = nib.load(glob.glob(os.path.join(self.segmentation_dir, "*ventricles*.nii.gz"))[0])
        wholebrain_data = wholebrain_seg.get_fdata()
        vent_data = vent_seg.get_fdata()
    
        # Subtract ventricles (make sure masks are binary)
        result_data = np.where((wholebrain_data > 0) & (vent_data == 0), 1, 0)
    
        # Step 4: Save result
        result_img = nib.Nifti1Image(result_data.astype(np.uint8), affine=wholebrain_seg.affine)
        result_out_fpath = os.path.join(self.segmentation_dir, "global_bin.nii.gz")
        nib.save(result_img, result_out_fpath)

    def run_surface_gen(self):
        """
        Run surface .stl generation
        """
        # Directories
        if self.parameters["segmentation_dir"] or self.parameters["segmentations"]:
            self.segmentation_dir = os.path.join(self.input_dir, "segmentations")
        else:
            self.segmentation_dir = os.path.join(self.output_dir, "segmentations")
            
        self.interim_dir = os.path.join(self.interim_dir, "surface_generation")
        os.makedirs(self.interim_dir, exist_ok=True)

        # Freesurfer setup
        self.freesurfer_source = "source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        self.freesurfer_env = os.environ.copy()
        self.freesurfer_env["SUBJECTS_DIR"] = self.interim_dir
        
        # Create other ROI geometries
        self.loggers.plugin_log(f"Creating region surface files")
        self.regions = self.parameters["regions"].split(",")
        for region in self.regions:
            self.prepare_geometry(region)

        # Create global seg and surface (wholebrain minus ventricles)
        self.loggers.plugin_log(f"Creating global surface file")
        if self.parameters["generate_global"]:
                self.generate_global_surface()
                self.prepare_geometry("global")