NAME = "NeuroMPET"
PARAMETERS = {
    "props_fpath": {
        "type": str,
        "default": "",
        "help": "Path to optional properties file containing additional parameters"
    },
    "mesh_dir": {
        "type": str,
        "default": "",
        "help": "Path to directory containing mesh .vtk files"
    },
    "outer_surface_fpath": {
        "type": str,
        "default": "",
        "help": "Path to outer surface (wholebrain) .stl file"
    },
    "dwi_dir": {
        "type": str,
        "default": "",
        "help": "Path to directory containing diffusion-weighted imaging files"
    },
    "cbf_dir": {
        "type": str,
        "default": "",
        "help": "Path to directory containing cerebral blood flow files"
    },
    "adjust_labels_dwi": {
        "type": bool,
        "default": False,
        "help": "If True, ROI labels are updated based on DWI FA (default: False)"
    },
    "adjust_outer_labels": {
        "type": bool,
        "default": False,
        "help": "If True, outer surface labels are adjusted based on surface triangles (default: False)"
    },
    "generate_cbf_map": {
        "type": bool,
        "default": False,
        "help": "If True, generates a scalar CBF map (default: False)"
    },
    "generate_fa_map": {
        "type": bool,
        "default": False,
        "help": "If True, generates a scalar FA map (default: False)"
    },
}