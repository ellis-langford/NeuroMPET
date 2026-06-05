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
        "help": "Path to a directory containing all surface files or "
                + "a list of comma-separated paths to surface files"
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
    "mesh_iterations": {
        "type": int,
        "default": 50,
        "help": "Maximum number of absolute edge-length values to attempt during meshing (default: 50)"
    },
    "generate_region_meshes": {
        "type": bool,
        "default": False,
        "help": "If True, generate regional meshes in addition to the global mesh (default: False)"
    }
}
