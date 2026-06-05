NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "mesh": {
        "type": str,
        "default": "",
        "help": "Path to global mesh file"
    },
    "surfaces": {
        "type": str,
        "default": "",
        "help": "Path to a directory containing all surface files or "
                + "a list of comma-separated paths to surface files"
    },
}
