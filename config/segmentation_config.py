NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "subject_id": {
        "type": str,
        "default": "subject999",
        "help": "SubjectID of data to be analysed"
    },
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
    "segmentation_mode": {
        "type": str,
        "default": "SynthSeg",
        "help": "Method of segmentation, options: FreeSurfer/SynthSeg (default: SynthSeg)"
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
        "default": False,
        "help": "Aids processing if subject has enlarged ventricles (default: False)"
    },
    "large_FOV": {
        "type": bool,
        "default": False,
        "help": "Aids processing if subject has a field of view > 256 (default: False)"
    },
    "use_gpu": {
        "type": bool,
        "default": False,
        "help": "If True, GPU is used to run SynthSeg (default: False)"
    },
}