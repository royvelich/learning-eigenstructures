import torch
import numpy as np
import igl
from typing import Union, List, Optional, Tuple
from collections import defaultdict

### Attempt to import svd batch method. If not provided, use default method
try:
    from torch_batch_svd import svd as batch_svd
except ImportError:
    print("torch_batch_svd not installed. Using torch.svd instead")
    batch_svd = torch.svd


def least_sq_with_known_values(A, b, known=None):
    """Solves the least squares problem minx ||Ax - b||2, where some values of x are known.
    Works by moving all known variables from A to b."""

    M, N = A.shape
    M2, K = b.shape
    assert M == M2, "A's first dimension must match b's second"

    if known is None:
        known = {}

    # Move to b
    for index, val in known.items():
        col = A[:, index]
        b -= torch.einsum("i,j->ij", col, val)

    # Remove from A
    unknown = [n for n in range(N) if n not in known]
    A_reduced = A[:, unknown]  # only continue with cols for unknowns

    # Use modern torch.linalg.lstsq (note: arguments are swapped compared to old torch.lstsq)
    result = torch.linalg.lstsq(A_reduced, b)
    x = result.solution

    # all unknown values have now been found. Now create the output tensor
    x_out = torch.zeros((N, K)).to(A.device)
    if known is not None:
        # Assign initially known values to x_out
        for index, val in known.items():
            x_out[index] = val

        # Assign initially unknown values to x_out
        x_out[unknown] = x

    return x_out


def cholesky_invert(A):
    """Invert matrix using Cholesky decomposition"""
    L = torch.cholesky(A)
    L_inv = torch.inverse(L)
    A_inv = torch.mm(L_inv.T, L_inv)
    return A_inv


def get_cot_weights_full_libigl(vertices, faces, device="cuda"):
    """Get cotangent weights using libigl, adapted from the provided implementation"""

    # Convert to numpy for libigl
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()

    # Compute cotangent Laplacian using libigl
    L_libigl = igl.cotmatrix(vertices_np, faces_np)

    # Convert to dense tensor and negate to get standard convention
    L_dense = torch.from_numpy(L_libigl.toarray()).to(dtype=vertices.dtype, device=device)

    # Negate to get standard Laplacian convention
    L_standard = -L_dense

    # Extract weights: w_ij = -L_ij (off-diagonal entries)
    W = torch.zeros_like(L_standard)
    # Create mask for off-diagonal entries
    mask = ~torch.eye(L_standard.shape[0], dtype=torch.bool, device=device)
    W[mask] = -L_standard[mask]

    return W


def get_one_ring_neighbors(faces, n_vertices):
    """Get one-ring neighbors from faces"""

    faces_np = faces.cpu().numpy()
    mapping = defaultdict(set)

    for f in faces_np:
        for j in [0, 1, 2]:  # for each vert in face
            i, k = (j + 1) % 3, (j + 2) % 3  # get 2 other vertices
            mapping[f[j]].add(f[i])
            mapping[f[j]].add(f[k])

    # Convert to dict of lists and ensure all vertices are included
    neighbors = {}
    for v in range(n_vertices):
        neighbors[v] = list(mapping[v]) if v in mapping else []

    return neighbors


def produce_cot_weights_nfmt(w_full, one_ring_neighbors, device="cuda"):
    """Convert full cotangent weights to nfmt format"""

    n_vertices = w_full.shape[0]
    max_neighbors = max(len(neighbors) for neighbors in one_ring_neighbors.values()) if one_ring_neighbors else 1

    w_nfmt = torch.zeros((n_vertices, max_neighbors), dtype=w_full.dtype, device=device)

    for i in range(n_vertices):
        neighbors = one_ring_neighbors[i]
        for n, j in enumerate(neighbors):
            w_nfmt[i, n] = w_full[i, j]

    return w_nfmt


def produce_idxs(n_vertices, one_ring_neighbors, device="cuda"):
    """Produce flattened index arrays for efficient computation"""

    ii, jj, nn = [], [], []

    for i in range(n_vertices):
        neighbors = one_ring_neighbors[i]
        for n, j in enumerate(neighbors):
            ii.append(i)
            jj.append(j)
            nn.append(n)

    ii = torch.tensor(ii, dtype=torch.long, device=device)
    jj = torch.tensor(jj, dtype=torch.long, device=device)
    nn = torch.tensor(nn, dtype=torch.long, device=device)

    return ii, jj, nn


def produce_edge_matrix_nfmt(verts, edge_shape, ii, jj, nn, device="cuda"):
    """Given vertices, produce edge matrix in nfmt format"""

    E = torch.zeros(edge_shape, dtype=verts.dtype, device=device)
    if len(ii) > 0:
        E[ii, nn] = verts[ii] - verts[jj]

    return E


class ARAPDeformer:
    """
    As-Rigid-As-Possible surface deformation using the optimized implementation
    from pytorch-arap, but with libigl cotangent weights and the same public interface.
    """

    def __init__(
            self,
            vertices: Union[torch.Tensor, np.ndarray],
            faces: Union[torch.Tensor, np.ndarray, None],
            device: Optional[torch.device] = None,
            precomputed_laplacian: Optional = None  # Ignored for interface compatibility
    ) -> None:
        """
        Initialize ARAP deformer.

        Args:
            vertices: Vertex positions (N, 3)
            faces: Face indices (F, 3) or None
            device: Torch device to use
            precomputed_laplacian: Ignored (kept for interface compatibility)
        """
        # Setup device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Convert and store original data
        self._setup_mesh_data(vertices, faces)

        # Initialize constraint lists
        self.handle_indices = []
        self.fixed_indices = []

        # Initialize current vertices to original
        self._current_vertices = self._original_vertices.clone()

        # Compute mesh topology and weights using the optimized approach
        self._setup_arap_data()

    def _setup_mesh_data(self, vertices, faces):
        """Convert inputs to tensors with consistent format"""

        # Convert vertices
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        if vertices.dtype == torch.float64:
            vertices = vertices.float()
        elif vertices.dtype not in [torch.float32, torch.float64]:
            vertices = vertices.float()

        self._original_vertices = vertices.to(self.device)
        self.n_vertices = vertices.shape[0]

        # Convert faces
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces)
            if faces.dtype != torch.long:
                faces = faces.long()
            self._faces = faces.to(self.device)
        else:
            self._faces = None

    def _setup_arap_data(self):
        """Setup ARAP data structures using the optimized approach"""

        if self._faces is None:
            # Handle point cloud case
            self.one_ring_neighbors = {i: [] for i in range(self.n_vertices)}
            self.w_nfmt = torch.ones((self.n_vertices, 1), dtype=self._original_vertices.dtype, device=self.device)
            self.max_neighbors = 1
            self.ii = torch.tensor([], dtype=torch.long, device=self.device)
            self.jj = torch.tensor([], dtype=torch.long, device=self.device)
            self.nn = torch.tensor([], dtype=torch.long, device=self.device)
            self.L = torch.eye(self.n_vertices, dtype=self._original_vertices.dtype, device=self.device)
            return

        # Get one-ring neighbors
        self.one_ring_neighbors = get_one_ring_neighbors(self._faces, self.n_vertices)

        # Compute cotangent weights using libigl
        w_full = get_cot_weights_full_libigl(self._original_vertices, self._faces, self.device)

        # Convert to nfmt format for efficiency
        self.w_nfmt = produce_cot_weights_nfmt(w_full, self.one_ring_neighbors, self.device)

        # Store useful constants
        self.max_neighbors = max(len(neighbors) for neighbors in self.one_ring_neighbors.values())

        # Precompute index arrays for efficient computation
        self.ii, self.jj, self.nn = produce_idxs(self.n_vertices, self.one_ring_neighbors, self.device)

        # Build Laplacian matrix (standard convention)
        self.L = torch.diag(torch.sum(w_full, dim=1)) - w_full

    def set_handles(self, handle_indices: Union[torch.Tensor, np.ndarray, List[int]]) -> None:
        """Set handle vertices that will be moved during deformation"""
        if isinstance(handle_indices, (torch.Tensor, np.ndarray)):
            handle_indices = handle_indices.tolist()
        self.handle_indices = list(handle_indices)

    def set_fixed(self, fixed_indices: Union[torch.Tensor, np.ndarray, List[int]]) -> None:
        """Set fixed vertices that will remain at their original positions"""
        if isinstance(fixed_indices, (torch.Tensor, np.ndarray)):
            fixed_indices = fixed_indices.tolist()
        self.fixed_indices = list(fixed_indices)

    def deform(
            self,
            target_positions: Union[torch.Tensor, np.ndarray],
            max_iterations: int = 10,
            tolerance: float = 1e-6
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform ARAP deformation using the optimized algorithm.

        Args:
            target_positions: Target positions for handle vertices
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for energy change

        Returns:
            (deformed_vertices, num_iterations)
        """
        if len(self.handle_indices) == 0:
            return self._current_vertices.clone(), 0

        # Convert target positions
        if isinstance(target_positions, np.ndarray):
            target_positions = torch.from_numpy(target_positions)
        if target_positions.dtype != self._original_vertices.dtype:
            target_positions = target_positions.to(dtype=self._original_vertices.dtype)
        target_positions = target_positions.to(self.device)

        if target_positions.shape[0] != len(self.handle_indices):
            raise ValueError(f"Expected {len(self.handle_indices)} target positions, got {target_positions.shape[0]}")

        # Setup constraints using the original implementation's approach
        known_handles = {i: pos for i, pos in zip(self.handle_indices, target_positions)}
        known_static = {v: self._original_vertices[v] for v in self.fixed_indices}
        known = {**known_handles, **known_static}

        # Initial guess using Naive Laplacian editing
        p_prime = least_sq_with_known_values(self.L, torch.mm(self.L, self._original_vertices), known=known)

        if max_iterations == 0:
            self._current_vertices = p_prime
            return self._current_vertices.clone(), 0

        # Prepare for iterations - following the original implementation
        unknown_verts = [n for n in range(self.n_vertices) if n not in known]

        # Precompute fixed terms
        b_fixed = torch.zeros((self.n_vertices, 3), device=self.device)
        for k, pos in known.items():
            b_fixed += torch.einsum("i,j->ij", self.L[:, k], pos)

        # Precompute reduced Laplacian inverse
        L_reduced = self.L[unknown_verts][:, unknown_verts]
        L_reduced_inv = cholesky_invert(L_reduced)

        # Edge matrix shape for efficient computation
        edge_shape = (self.n_vertices, self.max_neighbors, 3)
        P = produce_edge_matrix_nfmt(self._original_vertices, edge_shape, self.ii, self.jj, self.nn, self.device)

        # Main iteration loop - following the original implementation exactly
        prev_energy = float('inf')

        for iteration in range(max_iterations):
            # Compute edge matrix for current deformation
            P_prime = produce_edge_matrix_nfmt(p_prime, edge_shape, self.ii, self.jj, self.nn, self.device)

            # Calculate covariance matrices in bulk
            D = torch.diag_embed(self.w_nfmt, dim1=1, dim2=2)
            S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))

            # Handle unchanged vertices
            unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=2).all(dim=1))[0])
            S[unchanged_verts] = 0

            # SVD to compute rotations
            U, sig, W = batch_svd(S)
            R = torch.bmm(W, U.permute(0, 2, 1))

            # Ensure proper rotations
            entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()
            if len(entries_to_flip) > 0:
                Umod = U.clone()
                cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)
                Umod[entries_to_flip, :, cols_to_flip] *= -1
                R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

            # Build RHS of minimum energy equation
            Rsum_shape = (self.n_vertices, self.max_neighbors, 3, 3)
            Rsum = torch.zeros(Rsum_shape, device=self.device)
            if len(self.ii) > 0:
                Rsum[self.ii, self.nn] = R[self.ii] + R[self.jj]

            # Batch multiply for efficiency
            Rsum_batch = Rsum.view(-1, 3, 3)
            P_batch = P.view(-1, 3).unsqueeze(-1)

            # Compute RHS
            b = 0.5 * (self.w_nfmt[..., None] *
                       torch.bmm(Rsum_batch, P_batch).squeeze(-1).reshape(self.n_vertices, self.max_neighbors, 3)).sum(dim=1)

            b -= b_fixed  # subtract constraint component

            # Solve for unknown vertices
            p_prime_unknown = torch.mm(L_reduced_inv, b[unknown_verts])

            # Reconstruct full solution
            new_p_prime = torch.zeros_like(p_prime)
            for index, val in known.items():
                new_p_prime[index] = val
            new_p_prime[unknown_verts] = p_prime_unknown

            # Check convergence using energy
            current_energy = self._compute_energy(new_p_prime, R)
            if abs(prev_energy - current_energy) < tolerance:
                p_prime = new_p_prime
                break

            p_prime = new_p_prime
            prev_energy = current_energy

        self._current_vertices = p_prime
        return self._current_vertices.clone(), iteration + 1

    def _compute_energy(self, p_prime, R):
        """Compute ARAP energy following the original implementation"""

        edge_shape = (self.n_vertices, self.max_neighbors, 3)
        P = produce_edge_matrix_nfmt(self._original_vertices, edge_shape, self.ii, self.jj, self.nn, self.device)
        P_prime = produce_edge_matrix_nfmt(p_prime, edge_shape, self.ii, self.jj, self.nn, self.device)

        # Compute energy
        rot_rigid = torch.bmm(R, P.permute(0, 2, 1)).permute(0, 2, 1)
        stretch_vec = P_prime - rot_rigid
        stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)
        energy = (self.w_nfmt * stretch_norm).sum()

        return energy.item()

    @property
    def vertices(self) -> torch.Tensor:
        """Get current deformed vertices"""
        return self._current_vertices.clone()

    @property
    def faces(self) -> torch.Tensor:
        """Get face indices"""
        return self._faces.clone() if self._faces is not None else None

    @property
    def original_vertices(self) -> torch.Tensor:
        """Get original undeformed vertices"""
        return self._original_vertices.clone()