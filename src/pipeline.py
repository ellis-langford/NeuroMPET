# Imports
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Processing imports
from src.core.inputs import Inputs
from src.core.image_prep import ImagePrep
from src.core.registration import Registration
from src.core.cortical_seg import CorticalSeg
from src.core.surface_generation import SurfaceGen
from src.core.mesh_generation import MeshGen
from src.core.mesh_map import MeshMap
from src.core.solver import Solver
from src.core.results_processing import ResultsProcessor
from config.region_definitions import REGION_GROUPS

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
        meshgen_params = self.load_parameters(config_fpath="/app/config/meshgen_config.py")
        meshmap_params = self.load_parameters(config_fpath="/app/config/meshmap_config.py")
        modelling_params = self.load_parameters(config_fpath="/app/config/modelling_config.py")
        results_params = self.load_parameters(config_fpath="/app/config/results_config.py")

        # Combine parameter files
        self.parameters = (
            core_params
            | preprocessing_params
            | registration_params
            | segmentation_params
            | surfacegen_params
            | meshgen_params
            | meshmap_params
            | modelling_params
            | results_params
        )

        # Give the logger access to resolved runtime parameters.
        self.loggers.parameters = self.parameters
        self.region_definitions = self.flatten_region_definitions(REGION_GROUPS)

    def flatten_region_definitions(self, region_groups):
        """
        Flatten grouped region definitions into pipeline region definitions.
        """
        flat_definitions = {}
        for base_region, group_definition in region_groups.items():
            user_input = group_definition.get("user_input")
            both_definition = group_definition.get("both")

            for subsection, definition in group_definition.items():
                if subsection == "user_input":
                    continue

                region_name = base_region if subsection == "both" else f"{base_region}_{subsection}"
                flat_definition = definition.copy()
                if user_input is not None:
                    flat_definition["user_input"] = user_input
                if subsection in ("L", "R"):
                    flat_definition["side"] = subsection

                    # NextBrain writes side-specific files; inherit the atlas label
                    # from the overall region and binarise each side directly.
                    if (
                        both_definition
                        and both_definition.get("nextbrain_labels")
                        and not flat_definition.get("combine_regions")
                        and not flat_definition.get("subtract_regions")
                    ):
                        flat_definition["region_type"] = "segmentation"
                        flat_definition["nextbrain_labels"] = both_definition["nextbrain_labels"]

                flat_definitions[region_name] = flat_definition

        return flat_definitions

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

        # SynthSeg/FreeSurfer environment variables
        self.synthseg_env = os.environ.copy()
        self.synthseg_env["SUBJECTS_DIR"] = self.base_dir

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
            cortical_seg = CorticalSeg(input_prepper)
            cortical_seg.run_cortical_seg()

        # Generate surfaces
        if self.parameters["run_surface_generation"]:
            surface_gen = SurfaceGen(input_prepper)
            surface_gen.run_surface_gen()

        # Generate tetrahedral mesh
        if self.parameters["run_mesh_generation"]:
            mesh_gen = MeshGen(input_prepper)
            mesh_gen.run_mesh_gen()

        # Map mesh to obtain ROI labels and scalar maps
        if self.parameters["run_mesh_mapping"]:
            mapper = MeshMap(input_prepper)
            mapper.run_mapping()

        # MPET Modelling
        if self.parameters["run_modelling"]:
            modeller = Solver(input_prepper)
            modeller.run_modelling()

        # Post-process modelling outputs
        if self.parameters["run_results_processing"]:
            results_processor = ResultsProcessor(input_prepper)
            results_processor.run_results_processing()

        # Complete
        self.loggers.plugin_log(f"{self.config['NAME']} - Execution complete: {self.loggers.now_time()}")
        self.loggers.log_success()
