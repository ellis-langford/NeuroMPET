NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "use_gpu": {
        "type": bool,
        "default": False,
        "help": "If True, GPU is used to run atlas segmentations (default: False)"
    },
}
