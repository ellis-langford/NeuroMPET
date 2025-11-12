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
    "bc_fpath": {
        "type": str,
        "default": "",
        "help": "Path to boundary conditions (.csv) file"
    },
    "labels_fpath": {
        "type": str,
        "default": "",
        "help": "Path to ROI label file (required if --run_meshing is False)"
    },
    "timestep_size": {
        "type": float,
        "default": 0.1,
        "help": "Time-step size (default: 0.1)"
    },
    "timestep_count": {
        "type": int,
        "default": 10,
        "help": "Number of time steps per boundary condition waveform (default: 10)"
    },
    "timestep_interval": {
        "type": int,
        "default": 100,
        "help": "Interval between two VTU output files (default: 100)"
    },
}
