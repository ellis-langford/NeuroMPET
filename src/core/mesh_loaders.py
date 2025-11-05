# Imports
import os
import sys
import glob
import numpy as np
import meshio
from collections import defaultdict

class MeshLoaders(object):
    """Class setup"""
    def __init__(self, plugin_obj, **kwargs):
        """
        Instantiate
        """
        super().__init__(**kwargs)
        
    def read_txt(self, path, dtype=float, index_file=False, remove_idx_col=True):
        """
        Read txt file
        
        Parameters:
        ---
        path (str) : Path to .txt file to be read
        dtype (type) : Type to read in file contents as
        index_file (bool) : If True, file contents is treated as indices and converted to zero-index
        remove_idx_col (bool) : If True, index column (if present), is removed from the file
        """
        # Load file
        arr = np.loadtxt(path, dtype=dtype)

        if arr.ndim == 1:
            arr = arr[:, None]
    
        # Remove index column
        if remove_idx_col:
            first_col = arr[:, 0].astype(int)
            if np.all(first_col == np.arange(len(first_col))) or np.all(first_col == np.arange(1, len(first_col)+1)):
                arr = arr[:, 1:]
    
        # Convert to zero-index
        if index_file:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr - 1
            else:
                int_cols = np.all(arr == arr.astype(int), axis=0)
                arr[:, int_cols] = arr[:, int_cols].astype(int) - 1

        return arr

    def save_txt(self, path, data, fmt="%d", index_file=False, add_index_col=True):
        """
        Save numpy array to text file with optional index adjustments
        
        Parameters:
        ---
        path (str): Output file path
        data (np.array): Data to save
        fmt (str): Format string for np.savetxt
        index_file (bool): If True, adjusts for 1-indexing if index-based files
        add_index_col (bool): If True, adds an index column to the output
        """
        # Ensure data is np array
        arr = np.asarray(data)
    
        if arr.ndim == 1:
            arr = arr[:, None]
    
        # Convert to one-indexing for index files
        if index_file:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr + 1
            else:
                int_cols = np.all(arr == arr.astype(int), axis=0)
                arr[:, int_cols] = arr[:, int_cols].astype(int) + 1
    
        # Add index column if required
        if add_index_col:
            idxs = np.arange(1, len(arr) + 1)
            arr = np.column_stack([idxs, arr])

        # Save
        np.savetxt(path, arr, fmt=fmt)

    def extract_mesh_info(self, mesh_path, output_dir, region, cell_type="tetra"):
        """
        Extract information from mesh files to txt information files
    
        Parameters:
        ---
        mesh_path (str) : Path to directory containing .vtk mesh file
        output_dir (str) : Path to directory to save output .txt files to
        region (str) : Region that mesh belongs to, required for filename saving
        cell_type (str): Define whether input is mesh (tetra) or surface (triangles)
        """
        # Replace any "unsigned_char" dtypes with "int"
        if cell_type == "tetra":
            # Load file
            with open(mesh_path, "r") as f: 
                content = f.read()
            # Fix dtypes
            content = content.replace("unsigned_char", "int")
    
            # Save as txt
            base, _ = os.path.splitext(mesh_path)
            info_path = base + "_clean.vtk"
            with open(info_path, "w") as f: 
                f.write(content)

            mesh_path = info_path
        
        # Load mesh
        mesh = meshio.read(mesh_path)
        
        # Create output directory
        if region not in ["global", "outer_surface"]:
            output_dir = os.path.join(output_dir, region)
        os.makedirs(output_dir, exist_ok=True)
    
        # Node coordinates and connectivity
        node_coords = mesh.points
        tetra = None
        element_type = ["tetra", "tetra10"] if cell_type == "tetra" else ["triangle", "tri"]
        for cellblock in mesh.cells:
            if cellblock.type in element_type:
                tetra = cellblock.data
                break

        unique_points, inverse = np.unique(tetra.flatten(), return_inverse=True)
        tetra = inverse.reshape(tetra.shape)
        node_coords = node_coords[unique_points]

        # Node coordinates file
        np.savetxt(os.path.join(output_dir, f"{region}_node_coords.txt"), node_coords, fmt="%.6f")

        # Node indices file
        if cell_type == "tetra":
            np.savetxt(os.path.join(output_dir, f"{region}_tetra_indices.txt"), tetra, fmt="%d")
        else:
            np.savetxt(os.path.join(output_dir, f"{region}_face_indices.txt"), tetra, fmt="%d")
    
        # Node neighbours file for global mesh
        if region == "global":
            # Compute tetrahedron faces
            faces = np.vstack([
                np.sort(tetra[:, [0, 1, 2]], axis=1),
                np.sort(tetra[:, [0, 1, 3]], axis=1),
                np.sort(tetra[:, [0, 2, 3]], axis=1),
                np.sort(tetra[:, [1, 2, 3]], axis=1),
            ])
            # Tetrahedron IDs for each face
            tetra_ids = np.repeat(np.arange(len(tetra)), 4)
            
            # Sort vertex indices in each face
            faces_sorted = np.sort(faces, axis=1)
            
            # Build structured array for hashing and sorting
            dtype = np.dtype([("n0", faces_sorted.dtype),
                              ("n1", faces_sorted.dtype),
                              ("n2", faces_sorted.dtype)])
            faces_view = np.empty(faces_sorted.shape[0], dtype=dtype)
            faces_view["n0"] = faces_sorted[:, 0]
            faces_view["n1"] = faces_sorted[:, 1]
            faces_view["n2"] = faces_sorted[:, 2]
            
            # Sort faces to group identical together
            order = np.argsort(faces_view, order=("n0", "n1", "n2"))
            faces_sorted = faces_sorted[order]
            tetra_ids_sorted = tetra_ids[order]
            
            # Identify shared faces
            dupe_mask = np.all(faces_sorted[1:] == faces_sorted[:-1], axis=1)
            shared_face_indices = np.nonzero(dupe_mask)[0]
            
            # Build tetrahedron neighbour pairs
            t1 = tetra_ids_sorted[shared_face_indices]
            t2 = tetra_ids_sorted[shared_face_indices + 1]
            
            # Build symmetric adjacency
            all_pairs = np.vstack([np.column_stack([t1, t2]),
                                   np.column_stack([t2, t1])])
            
            # Sort and deduplicate
            all_pairs = np.unique(all_pairs, axis=0)
            
            # Construct adjacency list
            neighbours = defaultdict(list)
            for a, b in all_pairs:
                neighbours[a].append(b)
            
            # Convert to array form, padding with -1 (for boundary faces)
            max_neigh = max(len(v) for v in neighbours.values())
            tetra_neighbours = -np.ones((len(tetra), max_neigh), dtype=int)
            for k, v in neighbours.items():
                tetra_neighbours[k, :len(v)] = v
            
            # Save tetrahedral neighbours
            np.savetxt(os.path.join(output_dir, f"{region}_tetra_neighbours.txt"), tetra_neighbours, fmt="%d")

    def convert_to_vtk4_legacy(self, mesh_file):
        """
        Convert VTK 5.x unstructured grid file (with OFFSETS/CONNECTIVITY)
        into legacy CELLS format for compatibility with solver

        Parameters:
        ---
        mesh_file (str) : Path input mesh .vtk file to convert
        """
        # Read input file
        with open(mesh_file, "r") as f:
            lines = f.read().splitlines()

        if not any(line.strip().startswith("OFFSETS") for line in lines):
            return mesh_file
        else:
            out = []
            offsets = []
            connectivity = []
            num_cells = 0
            i = 0
        
            while i < len(lines):
                line = lines[i].strip()
        
                # Update header version
                if line.startswith("# vtk DataFile"):
                    out.append("# vtk DataFile Version 4.2")
        
                # Rebuild CELLS section
                elif line.startswith("CELLS"):
                    parts = line.split()
                    num_cells = int(parts[1])
                    i += 1
        
                    # Read OFFSETS section
                    i += 1
                    while not lines[i].startswith("CONNECTIVITY"):
                        offsets.extend(map(int, lines[i].split()))
                        i += 1
        
                    # Read CONNECTIVITY section
                    i += 1  # Skip "CONNECTIVITY ..." line
                    while not lines[i].startswith("CELL_TYPES"):
                        connectivity.extend(map(int, lines[i].split()))
                        i += 1
        
                    # Convert OFFSETS/CONNECTIVITY into legacy CELLS lines
                    cell_lines = []
                    prev = 0
                    for off in offsets:
                        cell_nodes = connectivity[prev:off]
                        cell_lines.append(f"{len(cell_nodes)} " + " ".join(map(str, cell_nodes)))
                        prev = off
                    total_ints = num_cells + len(connectivity)
                    out.append(f"CELLS {num_cells} {total_ints}")
                    out.extend(cell_lines)
        
                    # Skip to CELL_TYPES
                    continue
                else:
                    out.append(line)
                i += 1
        
            # Write output file
            outpath = os.path.join(os.path.dirname(mesh_file), "global_legacy.vtk")
            with open(outpath, "w") as f:
                f.write("\n".join(out))
    
        return outpath