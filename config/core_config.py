NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "input_im": {
        "type": str,
        "default": "",
        "help": "Path to the input image to be processed"
    },
    "verbose": {
        "type": bool,
        "default": False,
        "help": "If True, print detailed log messages for debugging purposes (default: False)"
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
        "help": "If True, the pipeline will run cortical segmentation (default: True)"
    },
    "run_surface_generation": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run surface generation (default: True)"
    },
    "run_mesh_generation": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run mesh generation (default: True)"
    },
    "run_mesh_mapping": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run mesh mapping (default: True)"
    },
    "run_modelling": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run MPET modelling (default: True)"
    },
    "run_results_processing": {
        "type": bool,
        "default": True,
        "help": "If True, the pipeline will run final results processing (default: True)"
    },
    "regions": {
        "type": list,
        "default": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "help": "List of user inputs, region labels, or region names to include for pipeline execution"
    },
}
