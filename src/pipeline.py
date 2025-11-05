# Imports
import os
import sys
import shutil

# Processing imports
from src.core.inputs import Inputs
from src.core.image_prep import ImagePrep
from src.core.registration import Registration
from src.core.cortical_seg import FreeSurfer
from src.core.surface_generation import SurfaceGen
from src.core.mesh_map import MeshMap
from src.core.solver import Solver

# Import custom utility modules
from utils.base_cog import BaseCog
from utils.utils import Utils
from utils.helpers import Loggers

# Pipeline class
class NeuroMPET(BaseCog):
    def __init__(self, **kwargs):
        """NeuroMPET class setup"""
        super().__init__(**kwargs)
        
        # Instantiate custom modules
        self.utils = Utils()
        self.loggers = Loggers()

        # Load parameters from CLI or properties file
        core_params = self.load_parameters(config_fpath="/app/config/core_config.py")
        preprocessing_params = self.load_parameters(config_fpath="/app/config/preprocessing_config.py")
        registration_params = self.load_parameters(config_fpath="/app/config/registration_config.py")
        segmentation_params = self.load_parameters(config_fpath="/app/config/segmentation_config.py")
        surfacegen_params = self.load_parameters(config_fpath="/app/config/surfacegen_config.py")
        meshmap_params = self.load_parameters(config_fpath="/app/config/meshmap_config.py")
        modelling_params = self.load_parameters(config_fpath="/app/config/modelling_config.py")

        # Combine parameter files
        self.parameters = (
            core_params
            | preprocessing_params
            | registration_params
            | segmentation_params
            | surfacegen_params
            | meshmap_params
            | modelling_params
        )

    def run_pipeline(self):
        """
        Run pipeline processing
        """
        self.loggers.plugin_log(f"{self.config['NAME']} - Starting execution: {self.loggers.now_time()}")

        # Tidy up log files
        self.loggers.tidy_up_logs()

        # Directories
        self.input_dir   = os.path.join(self.base_dir, "inputs")
        self.interim_dir = os.path.join(self.base_dir, "interim_outputs")
        self.log_dir     = os.path.join(self.base_dir, "logs")
        self.output_dir  = os.path.join(self.base_dir, "outputs")

        for _dir in [self.input_dir, self.interim_dir, 
                     self.log_dir, self.output_dir]:
            shutil.rmtree(_dir, ignore_errors=True)
            os.makedirs(_dir, exist_ok=True)

        # Record parameters
        self.loggers.log_options(self.parameters)

        # Prepare inputs
        input_prepper = Inputs(self)
        input_prepper.prepare_inputs()

        # Preprocess input image
        if self.parameters["run_preprocessing"]:
            preprocesser = ImagePrep(input_prepper)
            preprocesser.run_preprocessing()

        # Register input image
        if self.parameters["run_registration"]:
            registration = Registration(input_prepper)
            registration.run_registration()

        # Segment input image
        if self.parameters["run_cortical_segmentation"]:
            cortical_seg = FreeSurfer(input_prepper)
            cortical_seg.run_cortical_seg()

        # Segment input image
        if self.parameters["run_surface_generation"]:
            surface_gen = SurfaceGen(input_prepper)
            surface_gen.run_surface_gen()

        # Map meshes to obtain ROI labels and scalar maps
        if self.parameters["run_mesh_mapping"]:
            mapper = MeshMap(input_prepper)
            mapper.run_mapping()

        # MPET Modelling
        if self.parameters["run_modelling"]:
            modeller = Solver(input_prepper)
            modeller.run_modelling()

        # Complete
        self.loggers.plugin_log(f"{self.config['NAME']} - Execution complete: {self.loggers.now_time()}")
        self.loggers.log_success()