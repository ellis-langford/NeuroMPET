NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "run_preprocessing": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run image preprocessing (default: True)"
    },
    "run_registration": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run image registration (default: True)"
    },
    "run_cortical_segmentation": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run freesurfer cortical segmentation (default: True)"
    },
    "run_surface_generation": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run surface generation (default: True)"
    },
    "run_mesh_mapping": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run mesh mapping (default: True)"
    },
    "run_modelling": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run MPET modellling (default: True)"
    },
}