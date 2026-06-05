# Imports
import os
import sys
import ants
import nibabel as nib

class Registration(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributes are present
        to_inherit = [
            "loggers",
            "parameters",
            "input_dir",
            "interim_dir",
            "output_dir",
            "atlas",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def registration(
        self,
        reg_type: str,
        moving_im_path: str,
        fixed_im_path: str,
        moving_out_path: str,
        fixed_out_path: str = None,
    ):
        """
        Register a moving image to a fixed image.

        Parameters:
        ---
        reg_type (str)         : Type of registration.
        moving_im_path (str)   : Path to the moving image.
        fixed_im_path (str)    : Path to the fixed image.
        moving_out_path (str)  : Path to save the transformed moving image.
        fixed_out_path (str)   : Optional path to save a fixed image copy.
        """
        try:
            fixed_image  = ants.image_read(fixed_im_path)
            moving_image = ants.image_read(moving_im_path)
            self.loggers.verbose_log(
                f"Registering moving image {moving_im_path} to fixed image {fixed_im_path} using {reg_type}"
            )

            # Registration
            registration = ants.registration(fixed=fixed_image, 
                                             moving=moving_image, 
                                             type_of_transform=reg_type)
            # Save transform
            transform_fpath = registration["fwdtransforms"][0]

        except Exception as e:
            self.loggers.errors(f"Error in registration: {e}")

        # Apply transformation to the moving image
        transformed_moving = ants.apply_transforms(fixed=fixed_image,
                                                   moving=moving_image,
                                                   transformlist=[transform_fpath],
                                                   interpolator="welchWindowedSinc")
        
        # Save moving image
        ants.image_write(transformed_moving, moving_out_path)

        # Optionally copy the fixed image to an output path
        if fixed_out_path:
            ants.image_write(fixed_image, fixed_out_path)

        if not os.path.isfile(moving_out_path):
            self.loggers.errors("Registration of images failed")
    
    def revert_intensities(self, orig_im: str, reg_im: str, outpath: str) -> bool:
        """
        Restore the registered image intensity range to match the input image.

        Parameters:
        ---
        orig_im (str)      : Path to the original NIfTI image.
        reg_im (str)       : Path to the registered image to be intensity-reverted.
        outpath (str)      : Path to save the intensity-reverted image.

        Returns:
        ---
        bool : True if intensity reversion was performed, False if not needed.
        """
        self.loggers.verbose_log("Reverting image intensities")

        # Load images and extract data
        input_data = nib.load(orig_im).get_fdata()
        registered_im = nib.load(reg_im)
        registered_data = registered_im.get_fdata()

        if (input_data.min() != registered_data.min() or
            input_data.max() != registered_data.max()):
            # Calculate image intensity ranges
            input_data_range = input_data.max() - input_data.min()
            registered_data_range = registered_data.max() - registered_data.min()

            if input_data_range != 0 and registered_data_range != 0:
                # Apply scaling to match the input image's intensity range
                scaled_registered_data = ((registered_data - registered_data.min()) / 
                                          registered_data_range)
                
                scaled_registered_data *= input_data_range 
                scaled_registered_data += input_data.min()
            
                # Save reverted image
                reverted_im = nib.Nifti1Image(scaled_registered_data,
                                              affine=registered_im.affine)
                nib.save(reverted_im, outpath)
                return True

        return False

    def register_images(self):
        """
        Perform registration and write the registered image.
        """
        # Register
        fixed_im, moving_im = self.atlas, self.input_im
        moving_outpath = os.path.join(self.interim_dir, os.path.basename(moving_im))
        self.registration(self.reg_type, 
                          moving_im, fixed_im, 
                          moving_outpath)

        # Check registered outputs have been produced
        if not os.path.exists(moving_outpath):
            self.loggers.errors(f"Registered moving image has not been produced")

        # Revert moving image intensity range
        outpath = os.path.join(self.output_dir, "image.nii.gz")
        intensity_changed = self.revert_intensities(moving_im, moving_outpath, outpath)

        # If no change to intensity range, copy as is
        if not intensity_changed:
            nib.save(nib.load(moving_outpath), outpath)

        # Check final registered output has been produced
        if not os.path.exists(outpath):
            self.loggers.errors("Registered output image has not been produced")

    def run_registration(self):
        """
        Run registration of input image to MNI atlas.
        """
        # Define input image
        if self.parameters["run_preprocessing"]:
            self.input_im = os.path.join(self.output_dir, "image.nii.gz")
        else:
            self.input_im = os.path.join(self.input_dir, "image.nii.gz")

        # Define registration type
        self.reg_type = self.parameters["reg_type"]

        # Define processing directory
        self.interim_dir = os.path.join(self.interim_dir, "registration")
        os.makedirs(self.interim_dir, exist_ok=True)
        
        # Register
        self.loggers.plugin_log("Registering input image to MNI space")
        self.register_images()
