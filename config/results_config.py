NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "modelling_outputs": {
        "type": str,
        "default": "",
        "help": "Path to a directory containing modelling VTU output files"
    },
    "global_mask": {
        "type": str,
        "default": "",
        "help": "Path to global binary mask (required if --run_cortical_segmentation is False)"
    },
    "labels_file": {
        "type": str,
        "default": "",
        "help": "Path to detailed ROI region-label file"
    },
    "results_timestep": {
        "type": int,
        "default": 500,
        "help": "Solver timestep to use for results processing (default: 500)"
    },
    "volume_weighted_results": {
        "type": bool,
        "default": True,
        "help": "If True, compute regional results using tetra-volume weighting (default: True)"
    },
    "register_to_mni": {
        "type": bool,
        "default": False,
        "help": "If True, register the final image and result NIfTIs to the MNI atlas (default: False)"
    },
    "results_max_dist_mm": {
        "type": float,
        "default": 3.0,
        "help": "Maximum voxel-to-mesh distance for grid generation (default: 3.0)"
    },
}
