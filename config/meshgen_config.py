NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "surfaces": {
        "type": str,
        "default": "",
        "help": "Path to directory containing all surface files or" +
                "list of comma seperated paths to surface files"
    },
    "target_global_elements": {
        "type": int,
        "default": 2500000,
        "help": "Target number of elements for global mesh (default: 2_500_000)"
    },
    "tolerance": {
        "type": float,
        "default": 0.2,
        "help": "Tolerance for discrepancy between target elements and actual elements in mesh (default: 0.2)"
    },
    "coarseness_steps": {
        "type": int,
        "default": 10,
        "help": "Number of coarseness values to attempt during meshing (default: 10)"
    }
}