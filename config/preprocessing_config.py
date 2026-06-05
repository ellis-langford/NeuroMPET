NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "reset_origin": {
        "type": bool,
        "default": True,
        "help": "If True, the image origin is reset to the centre of the brain (default: True)"
    },
    "normalise_intensities": {
        "type": bool,
        "default": True,
        "help": "If True, image intensities are normalised (default: True)"
    },
    "rescale_max": {
        "type": int,
        "default": 1000,
        "help": "Desired maximum intensity value when rescaling image intensities (default: 1000)"
    },
    "n4_bias_correct": {
        "type": bool,
        "default": True,
        "help": "If True, perform N4 bias field correction using ANTs (default: True)"
    },
}
