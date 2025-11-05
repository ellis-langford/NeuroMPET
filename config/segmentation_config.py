NAME = "NeuroMPET"
PARAMETERS = {
    "input_im": {
        "type": str,
        "default": "",
        "help": "Path to the input image to be processed"
    },
    "freesurfer_outputs": {
        "type": str,
        "default": "",
        "help": "Path to freesurfer outputs directory (already processed inputs)"
    },
    "regions": {
        "type": str,
        "default": "cerebrum_L,cerebrum_R,cerebrumWM_L,cerebrumWM_R," +
                   "cerebellum_L,cerebellum_R,cerebellumWM_L,cerebellumWM_R," +
                   "brainstem_L,brainstem_R,ventricles,wholebrain",
        "help": "Comma seperated list of regions to process"
    },
    "big_vents": {
        "type": bool,
        "default": True,
        "help": "Aids processing if subject has enlarged ventricles (default: True)"
    },
    "large_FOV": {
        "type": bool,
        "default": False,
        "help": "Aids processing if subject has a field of view > 256 (default: False)"
    },
}