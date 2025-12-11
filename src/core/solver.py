# Imports
import os
import sys
import glob
import subprocess
from src.core.mesh_loaders import MeshLoaders

class Solver(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["loggers", "parameters", "base_dir", 
                      "input_dir", "interim_dir", "output_dir", 
                      "log_dir", "surface_dir", "mesh_dir",
                      "labels_file", "bc_file"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

        self.mesh_loader = MeshLoaders(self)

    def build_solver_command(self):
        """
        Build MPET solver command
        """
        # Define parameters from config
        self.timestep_size = self.parameters["timestep_size"]
        self.timestep_count = self.parameters["timestep_count"]
        self.timestep_interval = self.parameters["timestep_interval"]

        # Outdir
        self.modelling_outdir = os.path.join(self.output_dir, "modelling")
        os.makedirs(self.modelling_outdir, exist_ok=True)

        # Define command
        self.solver_command = f"make clean && make -s && " + \
                              f"./MPET3D '{self.bit_file}' '{self.modelling_outdir}' " + \
                              f"{self.timestep_size} {self.timestep_count} " + \
                              f"{self.timestep_interval} '{self.bc_file}' " + \
                              f"'{self.labels_file}'"

        # Define location of MPET source code
        self.source_code_dir = "/app/opt/mpet_source_code"

    def run_solver(self):
        """
        Run solver
        """
        self.loggers.plugin_log("Starting execution")
        self.solver_log = os.path.join(self.log_dir, "solver.log")
        self.loggers.plugin_log(f"Solver command: {self.solver_command}")
        with open(self.solver_log, "w") as outfile:
            solver_sub = subprocess.run(["bash", "-c",
                                            self.solver_command],
                                            cwd=self.source_code_dir,
                                            stdout=outfile,
                                            stderr=subprocess.STDOUT)
            
        if solver_sub.returncode != 0:
            self.loggers.errors(f"Solver execution returned non-zero exit status - " +
                                f"please check log file at {self.solver_log}")

        # Check required outputs have been produced
        for timestep in range(0, int(self.timestep_count * 50 / self.timestep_interval)):
            result_file = os.path.join(self.modelling_outdir, f"outputs_{timestep}.vtu")

            if not os.path.exists(result_file):
                self.loggers.errors(f"Solver has not produced an output file at timestep {timestep} " +
                                    f"- please check log file at {self.solver_log}")
        # Check regional file        
        if not os.path.exists(os.path.join(self.modelling_outdir, f"outputs_region.vtu")):
            self.loggers.errors(f"Solver has not produced a regional output file " +
                                f"- please check log file at {self.solver_log}")

    def run_modelling(self):
        """
        Running modelling processing
        """
        self.loggers.plugin_log("Running MPET solver")
        self.interim_dir = os.path.join(self.interim_dir, "modelling")
        os.makedirs(self.interim_dir, exist_ok=True)
        
        # Define meshes
        self.mesh_file = glob.glob(os.path.join(self.mesh_dir, "**", "global.vtk"), recursive=True)[0]
        
        # Define surface files
        self.wb_surface = glob.glob(os.path.join(self.surface_dir, "**", "*wholebrain*.stl"), recursive=True)[0]
        self.vent_surface = glob.glob(os.path.join(self.surface_dir, "**", "*ventricles*.stl"), recursive=True)[0]

        # Produce custom .bit file
        self.loggers.plugin_log("Creating custom .bit file")
        self.bit_file = os.path.join(self.input_dir, "global.bit")
        self.mesh_loader.convert_and_export_custom_mesh(self.mesh_file,
                                                        self.wb_surface,
                                                        self.vent_surface,
                                                        self.bit_file)
        
        # Produce inner and outer vtu files for visualisation
        self.mesh_loader.extract_surfaces_from_bit(self.bit_file,
                                                   self.interim_dir)

        # Run solver
        self.loggers.plugin_log("Running MPET solver")
        self.build_solver_command() # Build solver command
        self.run_solver() # Run Solver