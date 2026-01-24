# Imports
import os
import sys
import glob
import numpy as np
import nibabel as nib
import pyvista as pv
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, generate_binary_structure
from scipy.spatial import KDTree
from collections import deque

class SurfaceGen(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["loggers", "parameters", "base_dir", 
                      "input_dir", "interim_dir", "output_dir",
                      "log_dir", "segmentation_dir"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

    def segmentation_to_surface(self, segmentation, affine, smooth_sigma=0.7):
        """
        Converts binary segmentation to PyVista surface.

        Parameters:
        ---
        segmentation (np.array) : Data from binary segmentation
        affine (matrix) : Affine matrix for segmentation
        smooth_sigma (float) : Smoothing value to apply
        """
        data = gaussian_filter(segmentation.astype(float), sigma=smooth_sigma)
        data = (data - data.min()) / (data.max() - data.min())
        verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
        verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1))])
        verts_world = (affine @ verts_hom.T).T[:, :3]
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        
        surface = pv.PolyData(verts_world, faces_pv)
        surface = surface.connectivity(extraction_mode='largest') # Remove disconnected pieces
        surface = surface.clean()  # Remove duplicates and degenerate faces
        
        return surface

    def generate_global_surface(self):
        """
        Generate global surface:
        - Remove overseg of ventricles that lies outside wholebrain
        - Fill holes in wholebrain mask to ensure watertightness
        - Seperate overlaps between wholebrain and ventricles by selectively dilating
        - Generate wholebrain and ventricles stl surfaces
        - Generate global stl surface by combining wholebrain and ventricles
        """
        # Load wholebrain
        try:
            wb_bin = glob.glob(os.path.join(self.segmentation_dir, f"*wholebrain*.nii.gz"), recursive=True)[0]
        except:
            self.loggers.errors(f"A wholebrain segmentations must be provided if --generate_global is True")
        wb_data = nib.load(wb_bin).get_fdata().astype(bool)
        wb_affine = nib.load(wb_bin).affine

        # Load ventricles
        try:
            vent_bin = glob.glob(os.path.join(self.segmentation_dir, f"*ventricles*.nii.gz"), recursive=True)[0]
        except:
            self.loggers.errors(f"A ventricles segmentation must be provided if --generate_global is True")
        vent_data = nib.load(vent_bin).get_fdata().astype(bool)
        vent_affine = nib.load(vent_bin).affine
        
        ### Remove ventricular overseg ###
        vent_data = vent_data & wb_data
        
        ### Fill holes in wholebrain ###
        visited = np.zeros_like(wb_data, dtype=bool) # Create a mask of visited voxels
        q = deque() # Initialise queue with the 8 corners of the volume
        shape = wb_data.shape
        corners = [(0,0,0), (0,0,shape[2]-1), (0,shape[1]-1,0), (0,shape[1]-1,shape[2]-1),
                   (shape[0]-1,0,0), (shape[0]-1,0,shape[2]-1), (shape[0]-1,shape[1]-1,0), (shape[0]-1,shape[1]-1,shape[2]-1)]
        for c in corners:
            if not wb_data[c]:
                q.append(c)
                visited[c] = True
        
        # 6-connectivity flood-fill
        neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in neighbors:
                nx, ny, nz = x+dx, y+dy, z+dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    if not wb_data[nx,ny,nz] and not visited[nx,ny,nz]:
                        visited[nx,ny,nz] = True
                        q.append((nx,ny,nz))
        
        # All voxels not reachable from outside are interior holes
        interior_holes = (~visited) & (~wb_data)
        wb_data[interior_holes] = True # Fill them

        # Close diagonal pinholes & single-voxel gaps
        struct26 = generate_binary_structure(3, 2)
        wb_data = binary_closing(wb_data, structure=struct26, iterations=1)

        ### Separate touching surfaces ###
        radius=2
        max_iter = 5
        g = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1] # Spherical structuring element
        struct = (g[0]**2 + g[1]**2 + g[2]**2) <= radius**2

        # Try several times to seperate
        for i in range(max_iter):
            # Detect touching voxels
            touching = vent_data & wb_data
            if not np.any(touching):
                break
            # Dilate the touching region
            localised_growth = binary_dilation(touching, structure=struct)
            wb_data |= localised_growth # Grow wholebrain mask locally

        # Pad volumes to prevent open boundaries in marching cubes
        pad = 1
        wb_data = np.pad(wb_data, pad, mode="constant", constant_values=False)
        vent_data = np.pad(vent_data, pad, mode="constant", constant_values=False)
    
        wb_affine = wb_affine.copy()
        wb_affine[:3, 3] -= pad * np.diag(wb_affine)[:3]
        vent_affine = wb_affine
        
        ### Generate surfaces ###
        wb_surf = self.segmentation_to_surface(wb_data, wb_affine)
        vent_surf = self.segmentation_to_surface(vent_data, vent_affine)
        
        ### Check for overlaps and fix at surface level ###
        tolerance = 0.1
        max_iter = 5
        tree = KDTree(wb_surf.points)
        dists, idx = tree.query(vent_surf.points)
        
        if np.any(dists < tolerance):    
            # Ensure normals exist
            if wb_surf.point_normals is None:
                wb_surf.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    
            iteration = 0
            while iteration < max_iter:
                # Find closest cortical points to ventricle points
                vent_points = vent_surf.points
                tree = KDTree(wb_surf.points)
                distances, idx = tree.query(vent_points)
                distances = np.array(distances)
                idx = np.array(idx)
                
                # Identify points that are too close
                touching_mask = distances < tolerance
                if not np.any(touching_mask):
                    # No overlaps left
                    break
                
                # Compute minimal displacement
                displacement = (tolerance - distances[touching_mask])[:, np.newaxis]
                normals = wb_surf.point_normals[idx[touching_mask]]
                wb_surf.points[idx[touching_mask]] += normals * displacement
                
                iteration += 1
            
            # Check again for overlaps
            tree = KDTree(wb_surf.points)
            distances, _ = tree.query(vent_surf.points)
            if np.any(distances < 0.1):
                self.loggers.errors(f"Global surface generation failed - overlaps present")

        # Final watertightness surface repair
        if not wb_surf.is_manifold:
            wb_surf = wb_surf.clean(
                tolerance=1e-6,
                remove_non_manifold_edges=True,
                remove_degenerate_cells=True,
                inplace=False,
            )

        # Check watertight
        if not wb_surf.is_all_triangles:
            wb_surf.triangulate(inplace=True)
        if not wb_surf.is_manifold:
            self.loggers.errors(f"Wholebrain surface generation failed - not watertight")
        if not vent_surf.is_all_triangles:
            vent_surf.triangulate(inplace=True)
        if not vent_surf.is_manifold:
            self.loggers.errors(f"Ventricle surface generation failed - not watertight")

        # Save wholebrain and vent surfaces
        wb_surf.save(os.path.join(self.output_dir, "surfaces", f"wholebrain.stl"))
        vent_surf.save(os.path.join(self.output_dir, "surfaces", f"ventricles.stl"))

        # Save to .stl file
        global_surf = wb_surf.merge(vent_surf, merge_points=False)
        outpath = os.path.join(self.output_dir, "surfaces", f"global.stl")
        global_surf.save(outpath)
        
        if not os.path.exists(outpath):
            self.loggers.errors(f"Global surface generation failed")

    def generate_region_surface(self, region):
        """
        Generate regional surface files

        Parameters:
        ---
        region (str) : Region to generate surface of
        """
        # Global composite
        if region not in ["global", "wholebrain", "ventricles"]:            
            # Load data
            bin_data = glob.glob(os.path.join(self.segmentation_dir, f"*{region}*.nii.gz"))[0]
            data = nib.load(bin_data).get_fdata()
            affine = nib.load(bin_data).affine

            surface = self.segmentation_to_surface(data, affine, smooth_sigma=0.7)
    
            outpath = os.path.join(self.output_dir, "surfaces", f"{region}.stl")
            surface.save(outpath)
    
            if not os.path.exists(outpath):
                self.loggers.errors(f"Surface generation failed for {region}")

    def run_surface_gen(self):
        """
        Run surface .stl generation
        """
        # Directories            
        self.interim_dir = os.path.join(self.interim_dir, "surface_generation")
        os.makedirs(self.interim_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "surfaces"), exist_ok=True)
        
        
        # Generate global surface
        if self.parameters["generate_global"]:
            self.loggers.plugin_log(f"Creating global surface file")
            self.generate_global_surface()

        # Create ROI geometries
        self.loggers.plugin_log(f"Creating region surface files")
        self.regions = self.parameters["regions"].split(",")
        for region in self.regions:
            self.generate_region_surface(region)