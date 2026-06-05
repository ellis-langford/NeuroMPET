# Imports
import os
import sys
import glob
import shutil
import numpy as np
import pyvista as pv
import pytetwild


class MeshGen(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributes are present
        to_inherit = [
            "loggers",
            "parameters",
            "interim_dir",
            "output_dir",
            "surface_dir",
            "regions",
        ]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def count_tetrahedra(self, tet_mesh: pv.UnstructuredGrid) -> int:
        """
        Count tetrahedral cells in a PyVista UnstructuredGrid.

        Parameters:
        ---
        tet_mesh (pv.UnstructuredGrid) : The mesh to count tetrahedra in.

        Returns:
        ---
        int : The number of tetrahedral cells in the mesh.
        """
        # Check if 'celltypes' exists and is not empty
        celltypes = getattr(tet_mesh, "celltypes", None)
        if celltypes is None:
            return int(tet_mesh.n_cells)

        celltypes = np.asarray(celltypes)
        if celltypes.size == 0:
            return 0

        # Count cells of type VTK_TETRA (10)
        vtk_tetra = 10
        tetra_count = int(np.sum(celltypes == vtk_tetra))

        return tetra_count if tetra_count > 0 else int(tet_mesh.n_cells)

    def resolve_surface_path(self, region: str) -> str:
        """
        Resolve the exact surface file path for a region.

        Parameters:
        ---
        region (str)   : Region name to resolve.

        Returns:
        ---
        str : File path to the region STL surface.
        """
        # Find matching STL file for the region
        matches = glob.glob(
            os.path.join(self.surface_dir, "**", f"{region}.stl"),
            recursive=True,
        )

        # Check for no matches or multiple matches
        if len(matches) == 0:
            self.loggers.errors(
                f"Surface file {region}.stl does not exist in {self.surface_dir}"
            )

        if len(matches) > 1:
            self.loggers.errors(
                f"Multiple surface files matched {region}.stl in {self.surface_dir}: {matches}"
            )

        return matches[0]

    def generate_mesh(
        self,
        stl_file: str,
        outpath: str = None,
        edge_length_abs: float = 2.0,
        edge_length_fac: float = 0.05,
        epsilon: float = 1e-3,
        optimise: bool = True,
        coarsen: bool = False,
        simplify: bool = True,
        num_threads: int = 0,
    ) -> pv.UnstructuredGrid:
        """
        Generate a tetrahedral mesh from an STL surface.

        Parameters:
        ---
        stl_file (str)          : Path to the input STL surface.
        outpath (str)           : Optional path to save the generated mesh.
        edge_length_abs (float) : Absolute target tetrahedral edge length.
        edge_length_fac (float) : Relative target edge length as a fraction of bounding-box diagonal.
        epsilon (float)         : Envelope size relative to bounding-box diagonal.
        optimise (bool)         : If True, optimise the generated mesh.
        coarsen (bool)          : If True, coarsen the generated mesh where possible.
        simplify (bool)         : If True, simplify the input surface before meshing.
        num_threads (int)       : Number of threads to use. Zero uses all available cores.

        Returns:
        ---
        pv.UnstructuredGrid : The generated tetrahedral mesh.
        """
        # Mesh generation
        try:
            # Read STL as PyVista PolyData
            surf = pv.read(stl_file)
            if surf.n_cells == 0:
                self.loggers.errors(f"No cells found in {stl_file}")

            # Tetrahedralise with pytetwild
            tet = pytetwild.tetrahedralize_pv(
                surf,
                edge_length_abs=edge_length_abs,
                edge_length_fac=edge_length_fac,
                epsilon=epsilon,
                optimize=optimise,
                simplify=simplify,
                coarsen=coarsen,
                num_threads=num_threads,
            )
        except Exception as e:
            self.loggers.errors(f"Mesh generation failed for {stl_file}: {e}")

        # Save mesh if outpath provided
        if outpath:
            try:
                tet.save(outpath)
            except Exception as e:
                self.loggers.errors(f"Failed to save mesh to {outpath}: {e}")

        return tet

    def generate_global_mesh_with_search(self, stl_file: str) -> tuple[pv.UnstructuredGrid, float, int]:
        """
        Generate the global mesh and adjust edge length until the element
        count falls within tolerance.

        Parameters:
        ---
        stl_file (str)   : Path to the global STL surface.

        Returns:
        ---
        tuple[pv.UnstructuredGrid, float, int] : The generated global mesh, the selected edge length, 
                                                 and the final tetrahedral count.
        """
        # Get target element count and tolerance from parameters
        target_tets = int(self.parameters["target_global_elements"])
        tolerance = float(self.parameters.get("tolerance", 0.2))
        max_iters = int(self.parameters.get("mesh_iterations", 50))

        # Initial edge length and bounds for search
        edge_length_abs = 2.0
        min_edge_length_abs = 0.1
        lower_bound = target_tets * (1.0 - tolerance)
        upper_bound = target_tets * (1.0 + tolerance)

        # Iteratively adjust edge length to find a mesh with element count within bounds
        for attempt in range(1, max_iters + 1):
            self.loggers.verbose_log(
                f"Global meshing iteration {attempt}"
            )
            tet = self.generate_mesh(
                stl_file=stl_file,
                edge_length_abs=edge_length_abs,
            )
            n_tets = self.count_tetrahedra(tet)
            self.loggers.verbose_log(
                f"Global mesh tetrahedra={n_tets}"
            )

            # Check if the number of tetrahedra is within the specified bounds
            if lower_bound <= n_tets <= upper_bound:
                return tet, edge_length_abs, n_tets

            # Adjust edge length based on whether we have too many or too few elements
            if n_tets > upper_bound:
                edge_length_abs += 0.1
            else:
                edge_length_abs = max(min_edge_length_abs, edge_length_abs - 0.1)

        # If we exhaust all attempts without finding a suitable mesh, log an error
        self.loggers.errors(
            "Unable to generate a global mesh within tolerance after "
            f"{max_iters} attempts"
        )

    def run_mesh_gen(self):
        """
        Run tetrahedral mesh generation.
        """
        # Define directories
        self.interim_dir = os.path.join(self.interim_dir, "mesh_generation")
        os.makedirs(self.interim_dir, exist_ok=True)

        self.mesh_dir = os.path.join(self.output_dir, "meshes")
        os.makedirs(self.mesh_dir, exist_ok=True)

        # Generate global mesh
        self.loggers.plugin_log("Generating global mesh")
        global_surface_fpath = self.resolve_surface_path("global")
        global_mesh_outpath = os.path.join(self.mesh_dir, "global.vtk")
        self.global_mesh, selected_edge_length_abs, _ = self.generate_global_mesh_with_search(
            stl_file=global_surface_fpath,
        )
        try:
            self.global_mesh.save(global_mesh_outpath)
        except Exception as e:
            self.loggers.errors(f"Failed to save global mesh to {global_mesh_outpath}: {e}")

        # Generate region meshes
        if self.parameters["generate_region_meshes"]:
            self.loggers.plugin_log("Generating region meshes")
            regions = [
                region for region in self.regions
                if region not in ["global", "wholebrain", "ventricles"]
            ]
            for region in regions:
                surface_path = self.resolve_surface_path(region)
                outpath = os.path.join(self.mesh_dir, f"{region}.vtk")
                self.loggers.verbose_log(f"Generating mesh file for {region}")
                self.generate_mesh(
                    stl_file=surface_path,
                    outpath=outpath,
                    edge_length_abs=selected_edge_length_abs,
                )
                if not os.path.isfile(outpath):
                    self.loggers.errors(f"Regional mesh file not produced for {region}")

        if not os.path.isfile(global_mesh_outpath):
            self.loggers.errors("Global mesh file not produced")

        # Move tracked surface debug output into the mesh interim directory.
        for fname in ["__tracked_surface.stl", "_tracked_surface.stl"]:
            src = os.path.abspath(os.path.join(os.getcwd(), fname))
            if not os.path.isfile(src):
                continue

            dst = os.path.abspath(os.path.join(self.interim_dir, fname))
            if src == dst:
                continue

            try:
                shutil.copy2(src, dst)
                os.remove(src)
            except Exception as e:
                self.loggers.errors(f"Failed to move {fname} to {self.interim_dir}: {e}")
