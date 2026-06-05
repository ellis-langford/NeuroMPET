NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "input_atlas": {
        "type": str,
        "default": "/app/assets/mni_icbm152_atlas_t1.nii.gz",
        "help": "Path to atlas to register input image to (default: MNI atlas)"
    },
    "reg_type": {
        "type": str,
        "default": "Affine",
        "help": "Registration type to use - options: Affine/Rigid/SyN (default: Affine)"
    },
}
