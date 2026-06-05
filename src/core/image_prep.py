# Imports
import os
import sys
import shutil
import nibabel as nib
import numpy as np
import ants

class ImagePrep(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributes are present
        to_inherit = [
            "loggers",
            "parameters",
            "input_dir",
            "interim_dir",
            "output_dir",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def reset_origin(self, input_im: str, output_im: str):
        """
        Reset the image origin to the approximate centre of the brain.

        Parameters:
        ---
        input_im (str)      : Path to the input image to be reset.
        output_im (str)     : Path to save the origin-reset image.
        """
        # Load image
        nii = nib.load(input_im)
        data = nii.get_fdata()
        affine = nii.affine.copy()
        self.loggers.verbose_log("Resetting image origin")
        
        # Reset origin
        affine[0:3, 3] = 0
        
        # Centring based on image shape
        centre_shift = np.identity(4)
        centre_shift[0:3, 3] = -np.array(data.shape) / 2.0
        aff_out = affine @ centre_shift

        # Save image
        os.makedirs(os.path.dirname(output_im), exist_ok=True)
        nib.Nifti1Image(data.astype(np.float32), aff_out).to_filename(str(output_im))

        # Check required outputs have been produced
        if not os.path.exists(output_im):
            self.loggers.errors("Origin reset image has not been produced")
        else:
            self.input_im = output_im

    def n4_bias_correct(self, input_im: str, output_im: str, mask: object = None):
        """
        Perform N4 bias field correction.
    
        Parameters:
        ---
        input_im (str)      : Path to the input NIfTI image.
        output_im (str)     : Path to save the bias-corrected image.
        mask (object)       : Optional binary mask to constrain correction.
        """
        # Load the image
        img = ants.image_read(input_im)
        self.loggers.verbose_log("Running N4 bias correction")
    
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
        if not os.path.exists(output_im):
            self.loggers.errors("N4 bias corrected image has not been produced")
        else:
            self.input_im = output_im

    def normalise_intensities(self, input_im: str, output_im: str):
        """
        Normalise image intensities to a fixed maximum value.
        
        Parameters:
        ---
        input_im (str)      : Path to the input image.
        output_im (str)     : Path to save the intensity normalised image.
        """
        # Load image
        nii = nib.load(input_im)
        data = nii.get_fdata()
        affine = nii.affine.copy()
        rescale_max = self.parameters["rescale_max"]
        self.loggers.verbose_log("Normalising image intensities")

        # Check for zero max intensity to avoid division by zero
        if np.max(data) == 0:
            self.loggers.errors("Cannot normalise intensities for an image with maximum intensity 0")
            return

        # Rescale intensities to specified maximum value
        scale_factor = rescale_max / np.max(data)
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
        if not os.path.exists(output_im):
            self.loggers.errors("Intensity normalised image has not been produced")
        else:
            self.input_im = output_im

    def run_preprocessing(self):
        """
        Run the enabled preprocessing steps on the input image.
        """
        self.loggers.plugin_log("Preprocessing input image")

        # Define input image
        self.input_im = os.path.join(self.input_dir, "image.nii.gz")
        self.interim_dir = os.path.join(self.interim_dir, "preprocessing")
        os.makedirs(self.interim_dir, exist_ok=True)
        
        # Reset image origin
        if self.parameters["reset_origin"]:
            interim_outpath = os.path.join(self.interim_dir, "origin_reset", os.path.basename(self.input_im))
            self.reset_origin(self.input_im, interim_outpath)

        # N4 bias correction
        if self.parameters["n4_bias_correct"]:
            interim_outpath = os.path.join(self.interim_dir, "N4_corrected", os.path.basename(self.input_im))
            self.n4_bias_correct(self.input_im, interim_outpath)

        # Normalise image intensities
        if self.parameters["normalise_intensities"]:
            interim_outpath = os.path.join(self.interim_dir, "intensity_normed", os.path.basename(self.input_im))
            self.normalise_intensities(self.input_im, interim_outpath)
        
        # Copy the latest preprocessed image to the output directory
        shutil.copy(self.input_im, os.path.join(self.output_dir, "image.nii.gz"))
