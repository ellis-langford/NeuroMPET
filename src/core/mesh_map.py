# Imports
import os
import sys
import glob
import nibabel as nib
import numpy as np
import meshio
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from src.core.mesh_loaders import MeshLoaders
from collections import Counter, defaultdict

class MeshMap(object):
    """Class setup"""
    def __init__(self, plugin_obj):
        # Check all expected attributed are present
        to_inherit = ["loggers", "parameters", "base_dir", "input_dir", "mesh_dir", 
                      "surface_dir", "interim_dir", "output_dir", "log_dir"]
        for attr in to_inherit:
            try:
                setattr(self, attr, getattr(plugin_obj, attr))
            except AttributeError as e:
                print(f"Attribute Error - {e}")
                sys.exit(1)

        self.mesh_loader = MeshLoaders(self)

    def prepare_mesh_info_inputs(self):
        """
        Process meshes and surfaces into information txt files
        """
        # Extract global mesh information
        self.global_mesh = glob.glob(os.path.join(self.mesh_dir, "global", "*global*.vtk"))[0]
        self.global_info_dir = os.path.join(self.interim_dir, "global_mesh_info")
        self.mesh_loader.extract_mesh_info(self.global_mesh, self.global_info_dir, "global")

        # Extract regional mesh information
        regions = ["cerebrum_L", "cerebrum_R", "cerebrumWM_L", "cerebrumWM_R",
                   "cerebellum_L", "cerebellum_R", "cerebellumWM_L", "cerebellumWM_R",
                   "brainstem_L", "brainstem_R"]
        for region in regions:
            region_mesh = glob.glob(os.path.join(self.mesh_dir, region, f"*{region}*.vtk"))[0]
            self.regional_info_dir = os.path.join(self.interim_dir, "regional_mesh_info")
            self.mesh_loader.extract_mesh_info(region_mesh, self.regional_info_dir, region)

        # Extract outer surface stl information
        if self.parameters["surface_dir"]:
            outer_surface = glob.glob(os.path.join(self.surface_dir, "**", "*wholebrain*.stl"), recursive=True)[0]
            self.surface_info_dir = os.path.join(self.interim_dir, "surface_info")            
            self.mesh_loader.extract_mesh_info(outer_surface, self.surface_info_dir, "outer_surface", cell_type="triangle")

        # Load global files
        self.global_mesh_node_coords = self.mesh_loader.read_txt(os.path.join(self.global_info_dir, "global_node_coords.txt"), dtype=float) # Node coords
        self.global_mesh_tetra_indices = self.mesh_loader.read_txt(os.path.join(self.global_info_dir, "global_tetra_indices.txt"), dtype=int, index_file=True) # Tetrahedra node indices
        self.global_mesh_tetra_neighbours = self.mesh_loader.read_txt(os.path.join(self.global_info_dir, "global_tetra_neighbours.txt"), dtype=int, index_file=True) # Shared tetrahedra faces
        
    def classify_tetrahedra(self):
        """
        Classify which region each tetrahedral element belongs to
        """
        label_arrays = []
        region_files = []
    
        # Define regions
        self.regions = {
            "cerebrum": [1, 2],
            "cerebrumWM": [3, 4],
            "brainstem": [5, 6],
            "cerebellum": [7, 8],
            "cerebellumWM": [9, 10],
        }

        # Load node tree
        tree_global = KDTree(self.global_mesh_node_coords)

        # Loop over each region
        for region in self.regions:
            # Load region-specific coords and indices
            lh_node_coords   = self.mesh_loader.read_txt(os.path.join(self.regional_info_dir, f"{region}_L", f"{region}_L_node_coords.txt"), dtype=float)
            lh_tetra_indices = self.mesh_loader.read_txt(os.path.join(self.regional_info_dir, f"{region}_L", f"{region}_L_tetra_indices.txt"), dtype=int, index_file=True)
            rh_node_coords   = self.mesh_loader.read_txt(os.path.join(self.regional_info_dir, f"{region}_R", f"{region}_R_node_coords.txt"), dtype=float)
            rh_tetra_indices = self.mesh_loader.read_txt(os.path.join(self.regional_info_dir, f"{region}_R", f"{region}_R_tetra_indices.txt"), dtype=int, index_file=True)

            l_dists, _ = tree_global.query(lh_node_coords, k=1)
            r_dists, _ = tree_global.query(rh_node_coords, k=1)
            tol_L, tol_R = float(np.percentile(l_dists, 99)), float(np.percentile(r_dists, 99))

            # Build KD-trees
            tree_L = KDTree(lh_node_coords)
            tree_R = KDTree(rh_node_coords)

            labels = np.zeros(len(self.global_mesh_node_coords), dtype=np.int8)

            # Process in chunks to manage memory
            chunk_size = 500_000
            for start in range(0, len(self.global_mesh_node_coords), chunk_size):
                end = min(start + chunk_size, len(self.global_mesh_node_coords))
                chunk = self.global_mesh_node_coords[start:end]
        
                # Query distances
                dist_L, _ = tree_L.query(chunk, k=1)
                dist_R, _ = tree_R.query(chunk, k=1)
        
                # Assign labels
                labels_chunk = np.zeros(len(chunk), dtype=np.int8)

                # Within tolerance masks
                in_L = dist_L < tol_L
                in_R = dist_R < tol_R

                # Case 1: only in left
                labels_chunk[in_L & ~in_R] = self.regions[region][0]
                # Case 2: only in right
                labels_chunk[in_R & ~in_L] = self.regions[region][1]
                # Case 3: in both — pick whichever is closer
                both = in_L & in_R
                closer_to_L = dist_L[both] < dist_R[both]
                closer_to_R = ~closer_to_L

                both_idx = np.where(both)[0]
                labels_chunk[both_idx[closer_to_L]] = self.regions[region][0]
                labels_chunk[both_idx[closer_to_R]] = self.regions[region][1]

                labels[start:end] = labels_chunk
        
            # Save output
            output_file = os.path.join(self.interim_dir, "region_labels",  f"{region}_labels.txt")
            os.makedirs(os.path.join(self.interim_dir, "region_labels"), exist_ok=True)
            self.mesh_loader.save_txt(output_file, labels, add_index_col=False)

            label_arrays.append(labels)
            region_files.append(output_file)

        # Combine by taking the maximum label at each node
        n_nodes = len(label_arrays[0])
        combined_labels = np.zeros(n_nodes, dtype=int)
        
        for region, labels in zip(self.regions.keys(), label_arrays):
            mask = labels != 0
            combined_labels[mask] = labels[mask]

        # Identify labeled and unlabeled nodes
        unlabeled_idx = np.where(combined_labels == 0)[0]
        labeled_idx = np.where(combined_labels != 0)[0]
    
        # Build KD-tree on labeled nodes
        tree = KDTree(self.global_mesh_node_coords[labeled_idx])
    
        # Find nearest labeled node for each unlabeled one
        _, nearest = tree.query(self.global_mesh_node_coords[unlabeled_idx], k=1)
    
        # Assign the same label
        combined_labels[unlabeled_idx] = combined_labels[labeled_idx[nearest]]

        # Save combined labels
        output_file = os.path.join(self.interim_dir, "regional_node_labels.txt")
        self.mesh_loader.save_txt(output_file, combined_labels, add_index_col=False)
        self.labels_file = output_file

        # Log node information
        log_file = os.path.join(self.log_dir, "labelled_node_counts.txt")
        all_labels = np.unique(combined_labels)
        unlabelled = np.sum(combined_labels == 0)
        with open(log_file, "w") as f:
            f.write(f"Total global nodes: {n_nodes:,}\n")
            f.write(f"Unlabelled nodes:   {unlabelled:,} ({unlabelled / n_nodes:.2%})\n")
    
        for region, (left_idx, right_idx) in self.regions.items():
            n_left = np.sum(combined_labels == left_idx)
            n_right = np.sum(combined_labels == right_idx)
            total = n_left + n_right
            with open(log_file, "a") as f:
                f.write(f"{region:20s} L={n_left:,}  R={n_right:,}  Total={total:,}\n")

        # Check output file produced
        if not os.path.isfile(output_file):
            self.loggers.errors(f"Tetrahedra region label file not produced - {output_file}")
        else:
            self.labels_file = os.path.join(self.interim_dir, "regional_node_labels.txt")

    def node_to_cell_labels(self):
        """
        Convert node-based labels to cell-based labels using majority vote,
        with neighbor-based tie-breaking, and log summary statistics.
        """        
        # Load mesh
        mesh = meshio.read(self.global_mesh)
        tets = mesh.cells_dict["tetra"]
        n_cells = len(tets)
    
        # Load node labels
        node_labels = np.loadtxt(self.labels_file, dtype=int)
    
        # Initial cell labels (majority vote)
        cell_labels = np.zeros(n_cells, dtype=int)
        tie_cells = []
        for cid, tet in enumerate(tets):
            labels = node_labels[tet]
            labels_nonzero = labels[labels != 0]  # Ignore unlabelled nodes
            if len(labels_nonzero) == 0:
                cell_labels[cid] = 0
                continue
    
            counts = Counter(labels_nonzero)
            most_common_count = counts.most_common(1)[0][1]
            # All labels tied for most frequent
            tied_labels = [label for label, count in counts.items() if count == most_common_count]
            if len(tied_labels) == 1:
                cell_labels[cid] = tied_labels[0]
            else:
                # Neighbour-based assignment
                tie_cells.append(cid)
    
        # Build node -> cell mapping for neighbour lookup
        node_to_cells = defaultdict(list)
        for cid, tet in enumerate(tets):
            for n in tet:
                node_to_cells[n].append(cid)
    
        # Neighbour-based tie-breaking
        for cid in tie_cells:
            tet = tets[cid]
            neighbors = set()
            for n in tet:
                neighbors.update(node_to_cells[n])
            neighbors.discard(cid)
            neighbor_labels = [cell_labels[nid] for nid in neighbors if cell_labels[nid] != 0]
            if neighbor_labels:
                counts = Counter(neighbor_labels)
                most_common_count = counts.most_common(1)[0][1]
                tied = [label for label, count in counts.items() if count == most_common_count]
                cell_labels[cid] = min(tied)  # If still no consensus, revert to smallest label
            else:
                cell_labels[cid] = 0
    
        # Save cell labels file
        self.labels_file = os.path.join(self.interim_dir, "regional_cell_labels.txt")
        os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)
        with open(self.labels_file, "w") as f:
            for cid, label in enumerate(cell_labels, 1):  # 1-based indexing
                f.write(f"{cid} {label}\n")
    
        # Logging
        all_labels = np.unique(cell_labels)
        unlabelled = np.sum(cell_labels == 0)
        with open(os.path.join(self.log_dir, "labelled_cells_counts.txt"), "w") as f:
            f.write(f"Total global cells: {n_cells:,}\n")
            f.write(f"Unlabelled cells:   {unlabelled:,} ({unlabelled / n_cells:.2%})\n")

            for region, (left_idx, right_idx) in self.regions.items():
                n_left = np.sum(cell_labels == left_idx)
                n_right = np.sum(cell_labels == right_idx)
                total = n_left + n_right
                f.write(f"{region:20s} L={n_left:,}  R={n_right:,}  Total={total:,}\n")

    def create_mesh_with_labels(self):
        """
        Create a vtu file with attached ROI labels for visualisation
        """       
        # Load mesh
        mesh = meshio.read(self.global_mesh)
        cell_block = mesh.cells[0]
        n_cells = cell_block.data.shape[0]
        
        # Load cell labels
        raw = np.loadtxt(self.labels_file, dtype=int)
        
        # No index column
        if raw.ndim == 1:
            cell_labels = raw
        # Index column
        else:
            raw = raw[np.argsort(raw[:, 0])]
            cell_labels = raw[:, 1]
        
        # Attach ROI as cell-data
        mesh.cell_data["ROI"] = [cell_labels]
        
        # Save to file
        output_path = os.path.join(self.interim_dir, "global_with_labels.vtk")
        
        # Redirect stderr to suppress warnings
        stderr_orig = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            meshio.write(output_path, mesh, file_format="vtk", binary=False)
        finally:
            sys.stderr.close()
            sys.stderr = stderr_orig

    def revise_labels_by_dwi(self):
        """
        Revise mesh labels based on DWI FA values
        """
        # Load DWI images (tensor components, eigenvalues, FA, MD)
        nii_files = {
            "tensor": os.path.join(self.dwi_dir, "dwi_tensor.nii.gz"),
            "L1": os.path.join(self.dwi_dir, "dwi_L1.nii.gz"),
            "L2": os.path.join(self.dwi_dir, "dwi_L2.nii.gz"),
            "L3": os.path.join(self.dwi_dir, "dwi_L3.nii.gz"),
            "FA": os.path.join(self.dwi_dir, "dwi_FA.nii.gz"),
            "MD": os.path.join(self.dwi_dir, "dwi_MD.nii.gz"),
        }
        imgs = {k: nib.load(v).get_fdata().astype(np.float32) for k, v in nii_files.items()}
    
        # Compute inverse affine for voxel mapping (world -> voxel space)
        ref_img = nib.load(nii_files["FA"])
        inv_affine = np.linalg.inv(ref_img.affine)
    
        # Compute centroids of all global tetrahedra
        coords = self.global_mesh_node_coords[self.global_mesh_tetra_indices, :]
        centroids = coords.mean(axis=1)
    
        # Map centroids to voxel coordinates and clip to bounds
        homog_centroids = np.c_[centroids, np.ones(len(centroids))]
        voxel_coords = (inv_affine @ homog_centroids.T).T[:, :3]
        voxel_coords = np.round(voxel_coords).astype(int)
    
        dimX, dimY, dimZ = imgs["FA"].shape
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, dimX - 1)
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, dimY - 1)
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, dimZ - 1)
    
        # Initialise array to store image values per tetrahedron
        N = self.global_mesh_tetra_indices.shape[0]
        tetra_center = np.zeros((N, 11), dtype=np.float32)

        # Sample DWI tensor components into first six columns
        for j in range(6):
            tetra_center[:, j] = imgs["tensor"][voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2], j]

        # Sample L1, L2, L3, FA and MD into remaining columns
        tetra_center[:, 6] = imgs["L1"][voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2]]
        tetra_center[:, 7] = imgs["L2"][voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2]]
        tetra_center[:, 8] = imgs["L3"][voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2]]
        tetra_center[:, 9] = imgs["FA"][voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2]]
        tetra_center[:, 10] = imgs["MD"][voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2]]
    
        # Clean up negative values and invalid FA ranges
        for col in [0, 3, 5, 6, 7, 8, 10]:
            tetra_center[tetra_center[:, col] < 0, col] = 0
        tetra_center[tetra_center[:, 9] > 1, 9] = 0
    
        # Neighbor averaging to fill gaps (using sparse neighbour matrix)
        rows = self.global_mesh_tetra_neighbours[:, 0]
        cols = self.global_mesh_tetra_neighbours[:, 5]
        data = np.ones_like(rows, dtype=np.float32)
        W = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

        # Compute neighbour counts and mean values
        neighbour_counts = np.array(W.sum(axis=1)).flatten()
        neighbour_sums = W @ tetra_center
        neighbour_means = np.divide(
            neighbour_sums,
            neighbour_counts[:, None],
            out=np.zeros_like(neighbour_sums),
            where=neighbour_counts[:, None] > 0,
        )

        # Replace zero entries with neighbour means
        mask = (tetra_center == 0)
        tetra_center[mask] = neighbour_means[mask]
    
        # Replace any remaining zeros with adjusted global mean
        for col in range(tetra_center.shape[1]):
            zeros = np.where(tetra_center[:, col] == 0)[0]
            if zeros.size:
                mean_val = tetra_center[:, col].mean()
                tetra_center[zeros, col] = mean_val + (mean_val * len(zeros)) / (N - len(zeros))
    
        # Save output arrays to files (tensor and FA)
        self.tensor_file = os.path.join(self.interim_dir, "dwi_tensor.txt")
        self.FA_file = os.path.join(self.interim_dir, "dwi_FA.txt")
        self.mesh_loader.save_txt(self.tensor_file, tetra_center, fmt="%d " + "%f " * 11)
        self.mesh_loader.save_txt(self.FA_file, tetra_center[:, 9].reshape(-1,1), fmt="%d %f")
    
        # Load labels for adjustment
        c = self.mesh_loader.read_txt(self.tensor_file, dtype=np.float64, remove_idx_col=False)
        d = self.mesh_loader.read_txt(self.labels_file, dtype=int, remove_idx_col=False)
    
        # Extract grey matter left (GML) and grey matter right (GMR) FA values
        GML = c[d[:,1] == 1][:, [0, 10]]
        GMR = c[d[:,1] == 2][:, [0, 10]]
        aveGML = np.mean(GML[:,1], dtype=np.float64) * 2.1
        aveGMR = np.mean(GMR[:,1], dtype=np.float64) * 2.1
    
        # Update labels based on FA thresholds
        for i in range(d.shape[0]):
            if d[i,1] in [1,3] and c[i,10] < aveGML:
                d[i,1] = 1
            if d[i,1] in [2,4] and c[i,10] < aveGMR:
                d[i,1] = 2
    
        # Save adjusted labels
        output_file = os.path.join(self.interim_dir, "labels_fa_adjusted.txt")
        self.mesh_loader.save_txt(output_file, d, add_index_col=False)
        self.labels_file = output_file
    
        # Check output files were produced
        for fpath in [self.tensor_file, self.FA_file, output_file]:
            if not os.path.isfile(fpath):
                self.loggers.errors(f"File not produced - {fpath}")

    def revise_outer_tetra_labels(self):
        """
        Revise tetrahedral labels for tetrahedra that are part of the outer surface
        """
        # Load surface files
        self.out_surface_nodes = self.mesh_loader.read_txt(os.path.join(self.surface_info_dir, "outer_surface_node_coords.txt"), dtype=float)
        self.out_surface_faces = self.mesh_loader.read_txt(os.path.join(self.surface_info_dir, "outer_surface_face_indices.txt"), dtype=int, index_file=True)
        
        # Build KDTree for nearest neighbour matching
        tree = KDTree(self.global_mesh_node_coords)
        dists, matched = tree.query(self.out_surface_nodes)

        # Check node matching against tolerances
        max_tol = 1e-3
        if np.any(dists > max_tol):
            self.loggers.plugin_log(f"WARNING: Some outer surface nodes mapped > {max_tol}. Max dist = {dists.max():.4e}")

        # Remap face indices to global indices
        remapped = matched[self.out_surface_faces]

        # Save output file
        output_file = os.path.join(self.interim_dir, f"outer_surface_face_indices_mapped.txt")
        self.mesh_loader.save_txt(output_file, remapped, index_file=True, add_index_col=False)

        # Check file produced
        if not os.path.isfile(output_file):
            self.loggers.errors(f"Mapping surface face file not produced - {output_file}")
            
        # Load outer surface face indices and current labels
        outer_faces = self.mesh_loader.read_txt(os.path.join(self.interim_dir, "outer_surface_face_indices_mapped.txt"), dtype=int, index_file=True)
        labels = self.mesh_loader.read_txt(self.labels_file, dtype=int, remove_idx_col=False)
    
        # Sort vertices in each tetrahedron for consistent comparison
        tetra_sorted = np.sort(self.global_mesh_tetra_indices, axis=1)
        outer_sorted = np.sort(outer_faces, axis=1)
    
        # Add tetrahedron index as extra column for tracking
        tetra_sorted = np.hstack([tetra_sorted, np.arange(tetra_sorted.shape[0])[:, None]])
    
        # Sort rows lexicographically for fast intersection
        tetra_sorted = tetra_sorted[np.lexsort(tetra_sorted[:, ::-1].T)]
        outer_sorted = outer_sorted[np.lexsort(outer_sorted[:, ::-1].T)]
    
        # Helper: find intersecting rows by vertex combinations
        def intersect_rows(a, b):
            a = np.ascontiguousarray(a)
            b = np.ascontiguousarray(b)
            dtype = {"names": [f"f{i}" for i in range(a.shape[1])], "formats": a.shape[1]*[a.dtype]}
            return np.intersect1d(a.view(dtype), b.view(dtype), return_indices=True)[1]
    
        # Check all combinations of vertices
        rows_to_update = np.concatenate([
            intersect_rows(tetra_sorted[:, 0:3], outer_sorted), # First 3 vertices
            intersect_rows(tetra_sorted[:, 1:4], outer_sorted), # Last 3 vertices
            intersect_rows(tetra_sorted[:, [0,1,3]], outer_sorted), # Vertices 0,1,3
            intersect_rows(tetra_sorted[:, [0,2,3]], outer_sorted) # Vertices 0,2,3
        ])
    
        # Map back to original tetrahedron indices that need label updates
        tetra_indices = tetra_sorted[rows_to_update, 4].astype(int)
    
        # Update labels for outer face tetrahedra
        # 3&4 -> 1&2: cerebrum WM to cerebrum GM
        # 9&10 -> 7&8: cerebellum WM to cerebellum GM
        for old_label, new_label in zip([3, 4, 9, 10], [1, 2, 7, 8]):
            mask = labels[tetra_indices, 1] == old_label
            labels[tetra_indices[mask], 1] = new_label
    
        # Save revised labels
        output_file = os.path.join(self.interim_dir, "labels_outer_revised.txt")
        self.mesh_loader.save_txt(output_file, labels, add_index_col=False)

        # Check output file produced
        if not os.path.isfile(output_file):
            self.loggers.errors(f"Revised outer label file not produced - {output_file}")
        else:
            self.labels_file = output_file

    def map_scalar_to_tetra(self, nii_fpath, scalar_type):
        """
        Map a scalar field (e.g., CBF, FA) from a NIfTI volume onto the tetrahedral mesh

        Parameters:
        ---
        nii_fpath (str): Path to the scalar NIfTI volume
        scalar_type (str): Name of the scalar type for output file naming (e.g. "CBF" or "FA")
        """
        # Load scalar NIfTI
        nii = nib.load(nii_fpath)
        img = nii.get_fdata()
        dimX, dimY, dimZ = img.shape # Dimensions of volume
        affine = nii.affine # Affine transformation matrix
        inv_affine = np.linalg.inv(affine) # Inverse for world -> coordinate mapping
    
        # Compute average non-zero voxels in image
        mask_nonzero = img != 0
        average_val = img[mask_nonzero].mean()
    
        # Compute centroids of each tetrahedron in global mesh coordinates
        idx = self.global_mesh_tetra_indices
        coords = self.global_mesh_node_coords[idx, :3]
        centroids = coords.mean(axis=1)
    
        # Convert centroids from world coordinates to voxel coordinates
        homog_centroids = np.c_[centroids, np.ones(len(centroids))]
        voxel_coords = (inv_affine @ homog_centroids.T).T[:, :3]
        voxel_coords = np.round(voxel_coords).astype(int)
    
        # Clip voxel coordinates to image bounds
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, dimX - 1)
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, dimY - 1)
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, dimZ - 1)
    
        # Sample image values at tetrahedral centroids
        tetra_vals = img[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
    
        # Fill zeros using neighborhood average
        zero_indices = np.where(tetra_vals == 0)[0]
        for i in zero_indices:
            x, y, z = voxel_coords[i]
            xs = np.clip(np.arange(x-2, x+1), 0, dimX-1)
            ys = np.clip(np.arange(y-2, y+1), 0, dimY-1)
            zs = np.clip(np.arange(z-2, z+1), 0, dimZ-1)
            cube = img[np.ix_(xs, ys, zs)]
            cube_flat = cube.flatten()
            mask_nonzero_cube = cube_flat != 0
            n_nonzero = np.sum(mask_nonzero_cube)
            n_zero = cube_flat.size - n_nonzero

            # If all values are non-zero, use global average
            if n_nonzero == 0:
                tetra_vals[i] = average_val
            # Else, use weighted mean of non-zero and average
            else:
                tetra_vals[i] = (cube_flat[mask_nonzero_cube].mean() * n_nonzero + n_zero * average_val) / cube_flat.size
    
        # Save scalar values
        output_file = os.path.join(self.output_dir, f"{scalar_type}_map.txt")
        self.mesh_loader.save_txt(output_file, tetra_vals, fmt="%d %f")
    
        # Check output file produced
        if not os.path.isfile(output_file):
            self.loggers.errors(f"{scalar_type} scalar map file not produced - {output_file}")

    def run_mapping(self):
        """
        Run mesh mapping
        """
        self.interim_dir = os.path.join(self.interim_dir, "mesh_mapping")
        os.makedirs(self.interim_dir, exist_ok=True)

        self.dwi_dir = os.path.join(self.input_dir, "dwi_files")
        self.cbf_dir = os.path.join(self.input_dir, "cbf_files")
        
        regions = ["global", "ventricles", "brainstem_L", "brainstem_R",
                   "cerebrum_L", "cerebrum_R", "cerebrumWM_L", "cerebrumWM_R", 
                   "cerebellum_L", "cerebellum_R", "cerebellumWM_L", "cerebellumWM_R"]

        if self.parameters["run_mesh_mapping"]:
            # Create information files from meshes
            self.loggers.plugin_log("Preparing mesh information inputs")
            self.prepare_mesh_info_inputs()
    
            # Classify tetrahedra into regions
            self.loggers.plugin_log("Classifying tetrahedra into regions")
            self.classify_tetrahedra()
            self.loggers.plugin_log("Converting node labels to cell labels")
            self.node_to_cell_labels()
            self.create_mesh_with_labels()
    
            # Adjust labels based on FA
            if self.parameters["adjust_labels_dwi"]:
                self.loggers.plugin_log("Revising labels according to DWI FA values")
                self.revise_labels_by_dwi()
    
            # Revise outer tetrahedra labels
            if self.parameters["adjust_outer_labels"]:
                self.loggers.plugin_log("Adjusting outer tetrahedra labels")
                self.revise_outer_tetra_labels()
    
            # Produce CBF scalar map
            if self.parameters["generate_cbf_map"]:
                self.loggers.plugin_log("Generating CBF scalar map")
                cbf_file = glob.glob(os.path.join(self.cbf_dir, "*cbf*.nii*"))[0]
                self.map_scalar_to_tetra(cbf_file, "CBF")
    
            # Produce FA scalar map
            if self.parameters["generate_fa_map"]:
                self.loggers.plugin_log("Generating FA scalar map")
                fa_file = glob.glob(os.path.join(self.dwi_dir, "*FA*.nii*"))[0]
                self.map_scalar_to_tetra(fa_file, "FA")
    
            # Save final label file with index col
            labels = self.mesh_loader.read_txt(self.labels_file, dtype=int, remove_idx_col=False)
            self.mesh_loader.save_txt(os.path.join(self.output_dir, "labels.txt"), labels, add_index_col=True)