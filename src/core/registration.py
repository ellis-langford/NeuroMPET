# Imports
import os
import sys
import ants
import nibabel as nib

class Registration(object):
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

    def registration(self, reg_type, 
                     moving_im_path, fixed_im_path, 
                     moving_out_path, fixed_out_path):
        """
        Performs a registration between an image pair 

        Args:
            reg_type (str)            : Type of registration
            moving_image (str)        : Path to moving image
            fixed_image (str)         : Path to fixed image
            moving_out (str)          : Path to output transformed moving image
            fixed_out (str) : Path to output fixed image.
        """      
        try:
            fixed_image  = ants.image_read(fixed_im_path)
            moving_image = ants.image_read(moving_im_path)

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

        if not os.path.isfile(moving_out_path):
            self.loggers.errors("Registration of images failed")

    def register_images(self):
        """
        Performs image registration on input images
        """                
        # Register
        fixed_im, moving_im = self.atlas, self.input_im
        moving_outpath = os.path.join(self.interim_dir, os.path.basename(moving_im))
        fixed_outpath = os.path.join(self.interim_dir, os.path.basename(fixed_im))
        self.registration(self.reg_type, 
                          moving_im, fixed_im, 
                          moving_outpath, fixed_outpath)

        # Check registered outputs have been produced
        if not os.path.exists(moving_outpath):
            self.loggers.errors(f"Registered moving image has not been produced")

        # Revert moving image intensity range
        self.loggers.plugin_log("Reverting image intensities")
        outpath = os.path.join(self.output_dir, "registered_im.nii.gz")
        self.revert_intensities(moving_im, moving_outpath, outpath)

        # Check rescaled moving outputs have been produced
        if not os.path.exists(outpath):
            self.loggers.errors(f"Rescaled moving image has not been produced")
    
    def revert_intensities(self, orig_im, reg_im, outpath):
        """
        Revert post-registration image intensities
        """
        # Load images and extract data
        input_data = nib.load(orig_im).get_fdata()
        registered_im  = nib.load(reg_im)
        registered_data = registered_im.get_fdata()

        if (input_data.min() != registered_data.min() and
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

    def run_registration(self):
        """
        Run registration
        """
        # Define input image
        if self.parameters ["run_preprocessing"]:
            self.input_im = os.path.join(self.output_dir, "image.nii.gz")
        else:
            self.input_im = os.path.join(self.input_dir, "image.nii.gz")

        # Define atlas
        self.atlas = os.path.join(self.input_dir, "atlas.nii.gz")

        # Define registration type
        self.reg_type = self.parameters["reg_type"]

        # Define processing directory
        self.interim_dir = os.path.join(self.interim_dir, "registration")
        os.makedirs(self.interim_dir, exist_ok=True)
        
        # Register
        self.loggers.plugin_log("Running registration")
        self.register_images()