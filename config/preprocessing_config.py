NAME = "NeuroMPET"
PARAMETERS = {
    "input_im": {
        "type": str,
        "default": "",
        "help": "Path to the input image to be processed"
    },
    "modality": {
        "type": str,
        "default": "t1",
        "help": "Modality of the input image. Options: t1, t2, flair, bold, fa (default: t1)"
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
        "help": "Desired intensity of white matter peak when rescaling intensity (default: 1000)"
    },
    "n4_bias_correct": {
        "type": bool,
        "default": True,
        "help": "If True, perform N4 bias field correction using ANTs (default: True)"
    },
}