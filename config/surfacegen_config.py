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
        "help": "Path to a directory containing all segmentation files or "
                + "a list of comma-separated paths to segmentation files"
    }
}
