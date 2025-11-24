NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "meshes": {
        "type": str,
        "default": "",
        "help": "Path to directory containing all mesh files or" +
                "list of comma seperated paths to mesh files"
    },
    "dwi_dir": {
        "type": str,
        "default": "",
        "help": "Path to directory containing diffusion-weighted imaging files"
    },
    "cbf_dir": {
        "type": str,
        "default": "",
        "help": "Path to directory containing cerebral blood flow files"
    },
    "adjust_labels_dwi": {
        "type": bool,
        "default": False,
        "help": "If True, ROI labels are updated based on DWI FA (default: False)"
    },
    "generate_cbf_map": {
        "type": bool,
        "default": False,
        "help": "If True, generates a scalar CBF map (default: False)"
    },
    "generate_fa_map": {
        "type": bool,
        "default": False,
        "help": "If True, generates a scalar FA map (default: False)"
    },
}