# Imports
import os
import sys
import shutil
import nibabel as nib
import numpy as np
import ants
from scipy.signal import find_peaks

class ImagePrep(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["loggers", "parameters", "base_dir", 
                      "input_dir", "interim_dir", "output_dir"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def reset_origin(self, input_im, output_im):
        """
        Reset the origin of the image to the centre of the brain

        Params:
        ---
        input_im (str)        : Path to input image to be reset
        output_im (str)       : Path to save origin reset image to
        """
        # Load image
        nii = nib.load(input_im)
        data = nii.get_fdata()
        affine = nii.affine.copy()
        
        # Reset origin
        affine[0:3, 3] = 0
        
        # Centering based on image shape
        center_shift = np.identity(4)
        center_shift[0:3, 3] = -np.array(data.shape) / 2.0
        aff_out = affine @ center_shift

        # Save image
        os.makedirs(os.path.dirname(output_im), exist_ok=True)
        nib.Nifti1Image(data.astype(np.float32), aff_out).to_filename(str(output_im))

        # Check required outputs have been produced
        if not os.path.exists(output_im):
            self.loggers.errors(f"Origin reset image has not been produced")
        else:
            self.input_im = output_im

    def n4_bias_correct(self, input_im, output_im, mask=None):
        """
        Perform N4 bias field correction using ANTsPy.
    
        Params:
        ---
        input_im (str)                  : Path to the input NIfTI image.
        output_im (str)                 : Path to save the bias-corrected image.
        shrink_factor (int)             : Shrinking factor to speed up processing (default: 4).
        convergence (tuple)             : (max_iterations, convergence_threshold) (default: (50, 0.001)).
        mask (ants.ANTsImage, optional) : Optional binary mask to constrain correction.
        """
        # Load the image
        img = ants.image_read(input_im)
    
        # If no mask is provided, create one automatically
        if mask is None:
            mask = img.get_mask()
    
        # Apply N4 bias correction
        corrected = ants.n4_bias_field_correction(
            image=img,
            mask=mask,
            shrink_factor=4,
            convergence={"iters": [50], "tol": 0.001}
        )
    
        # Save output image
        os.makedirs(os.path.dirname(output_im), exist_ok=True)
        ants.image_write(corrected, output_im)

        # Check required outputs have been produced
        result_file = os.path.join(output_im)
        if not os.path.exists(result_file):
            self.loggers.errors(f"N4 bias corrected image has not been produced")
        else:
            self.input_im = output_im

    def normalise_intensities(self, input_im, output_im, rescale_max=1000):
        """
        Normalises image intensity so that the white matter peak maps to rescale_max.
        
        Params:
        ---
        input_im (str)    : Path to input image
        output_im (str)   : Path to save intensity-normalized image
        rescale_max (int) : Desired intensity value of white matter peak (default 1000)
        """
        scale_wm_peak = self.parameters["wm_peak_scaling"]
        # Load image
        nii = nib.load(input_im)
        data = nii.get_fdata()
        affine = nii.affine.copy()
    
        if scale_wm_peak:
            # Mask non-zero voxels (eps to account for non-zero background due to n4 correction)
            eps = 1e-3
            brain_voxels = data[data > eps]
            
            # Estimate white matter peak
            hist, bin_edges = np.histogram(brain_voxels, bins=1000, range=(np.min(brain_voxels), np.max(brain_voxels)))
            peaks, _ = find_peaks(hist)
    
            if len(peaks) == 0:
                self.loggers.errors("Could not find intensity peaks in histogram")
                return
    
            # Assume white matter peak is the highest peak (could refine further)
            wm_peak_index = peaks[np.argmax(hist[peaks])]
            wm_peak_intensity = bin_edges[wm_peak_index]
    
            # Scale image so that wm_peak_intensity → rescale_max
            scale_factor = rescale_max / wm_peak_intensity

        else:
            max_intensity = np.max(data)
            scale_factor = rescale_max / max_intensity
            
        norm_data = data * scale_factor
    
        # Clean up header
        new_header = nii.header.copy()
        new_header.set_data_dtype(np.float32)
        new_header["scl_slope"] = 1
        new_header["scl_inter"] = 0
    
        # Save image
        os.makedirs(os.path.dirname(output_im), exist_ok=True)
        nib.Nifti1Image(norm_data.astype(np.float32), affine, new_header).to_filename(str(output_im))
    
        # Check output
        result_file = os.path.join(output_im)
        if not os.path.exists(result_file):
            self.loggers.errors(f"White matter peak normalised image has not been produced")
        else:
            self.input_im = output_im

    def run_preprocessing(self):
        """
        Run image preprocessing
        """
        # Define input image
        self.input_im = os.path.join(self.input_dir, "image.nii.gz")
        self.interim_dir = os.path.join(self.interim_dir, "preprocessing")
        os.makedirs(self.interim_dir, exist_ok=True)
        
        # Reset image origin
        if self.parameters["reset_origin"]:
            self.loggers.plugin_log("Resetting image origin")
            interim_outpath = os.path.join(self.interim_dir, "origin_reset", os.path.basename(self.input_im))
            self.reset_origin(self.input_im, interim_outpath)

        # N4 Bias Correction
        if self.parameters["n4_bias_correct"]:
            self.loggers.plugin_log("Applying N4 Bias Correction")
            interim_outpath = os.path.join(self.interim_dir, "N4_corrected", os.path.basename(self.input_im))
            self.n4_bias_correct(self.input_im, interim_outpath)

        # Normalise image intensities
        if self.parameters["normalise_intensities"]:
            self.loggers.plugin_log("Normalising image intensity ranges")
            interim_outpath = os.path.join(self.interim_dir, "intensity_normed", os.path.basename(self.input_im))
            self.normalise_intensities(self.input_im, interim_outpath, self.parameters["rescale_max"])
        
        # Copy final image to output directory
        shutil.copy(self.input_im, os.path.join(self.output_dir, "preprocessed_im.nii.gz"))         