NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "segmentation_dir": {
        "type": str,
        "default": "",
        "help": "Path to directory containing all segmentation files"
    },
    "segmentations": {
        "type": str,
        "default": "",
        "help": "List of comma seperated paths to segmentations to process"
    },
    "regions": {
        "type": str,
        "default": "cerebrum_L,cerebrum_R,cerebrumWM_L,cerebrumWM_R," +
                   "cerebellum_L,cerebellum_R,cerebellumWM_L,cerebellumWM_R," +
                   "brainstem_L,brainstem_R,ventricles,wholebrain",
        "help": "Comma seperated list of regions to process"
    },
    "generate_global": {
        "type": bool,
        "default": True,
        "help": "If True, a global mesh will be created by subtracting ventricles from wholebrain (default: True)"
    }
}