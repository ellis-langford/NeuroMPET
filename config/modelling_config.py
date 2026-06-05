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
    "bc_file": {
        "type": str,
        "default": "",
        "help": "Path to boundary conditions (.csv) file"
    },
    "solver_labels_file": {
        "type": str,
        "default": "",
        "help": "Path to solver ROI label file (required if --run_mesh_mapping is False)"
    },
    "timestep_size": {
        "type": float,
        "default": 0.1,
        "help": "Size of timestep (default: 0.1)"
    },
    "waveform_timesteps": {
        "type": int,
        "default": 10,
        "help": "Number of time steps per boundary condition waveform (default: 10). "
                + "e.g. Waveform length = waveform_timesteps * timestep_size"
    },
    "num_waveforms": {
        "type": int,
        "default": 50,
        "help": "Number of total boundary condition waveforms to use to ensure steady-state reached (default: 50). "
                + "e.g. Total timesteps = waveform_timesteps * num_waveforms"
    },
    "output_timestep_interval": {
        "type": int,
        "default": 100,
        "help": "Interval between two VTU output files (default: 100)"
    },
}
