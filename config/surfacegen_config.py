NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "segmentations": {
        "type": str,
        "default": "",
        "help": "Path to directory containing all segmentation files or" +
                "list of comma seperated paths to segmentation files"
    },
    "generate_global": {
        "type": bool,
        "default": True,
        "help": "If True, a global mesh will be created by subtracting ventricles from wholebrain (default: True)"
    }
}