import numpy as np
import polyscope as ps
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from pyFM.mesh import TriMesh
import trimesh
import os
import argparse
from scipy.spatial.distance import pdist, squareform
import torch


def create_rbf_function(points, n_centers=10, rbf_type='thin_plate', seed=None):
    """Create a smooth function on points using radial basis functions."""
    if seed is not None:
        np.random.seed(seed)

    # Select random centers for the RBFs
    center_indices = np.random.choice(len(points), n_centers, replace=False)
    centers = points[center_indices]

    # Generate random weights for the RBFs
    weights = np.random.randn(n_centers)

    # Compute distances between points and centers
    distances = np.zeros((len(points), n_centers))
    for i in range(n_centers):
        distances[:, i] = np.sqrt(np.sum((points - centers[i]) ** 2, axis=1))

    # Define different RBF types
    if rbf_type == 'gaussian':
        # Gaussian RBF: exp(-r²)
        sigma = np.mean(distances) / 2  # Scale parameter
        rbf_values = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    elif rbf_type == 'multiquadric':
        # Multiquadric RBF: sqrt(1 + r²)
        epsilon = 1.0  # Shape parameter
        rbf_values = np.sqrt(1 + (epsilon * distances) ** 2)
    elif rbf_type == 'inverse_multiquadric':
        # Inverse multiquadric RBF: 1/sqrt(1 + r²)
        epsilon = 1.0  # Shape parameter
        rbf_values = 1.0 / np.sqrt(1 + (epsilon * distances) ** 2)
    elif rbf_type == 'thin_plate':
        # Thin plate spline RBF: r² log(r)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        rbf_values = np.zeros_like(distances)
        for i in range(n_centers):
            r = distances[:, i]
            mask = r > epsilon
            rbf_values[mask, i] = (r[mask] ** 2) * np.log(r[mask])
            # For small r, use Taylor expansion
            rbf_values[~mask, i] = 0
    else:
        raise ValueError(f"Unknown RBF type: {rbf_type}")

    # Compute the function values at each point
    function_values = rbf_values @ weights

    # Normalize to have unit L2 norm
    function_values = function_values / np.linalg.norm(function_values)

    return function_values


def create_sinusoidal_function(points, n_components=5, seed=None):
    """Create a smooth function on points using sinusoidal functions of the coordinates.

    Parameters:
    -----------
    points : array
        The point cloud coordinates of shape (n_points, 3).
    n_components : int
        Number of sinusoidal components to combine.
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    function_values : array
        Function values at each point, normalized to have unit L2 norm.
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Determine appropriate frequency range based on point cloud dimensions
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)

    # Start with zero function
    function_values = np.zeros(len(points))

    # Add sinusoidal components
    for i in range(n_components):
        # Random frequencies - scaled to be proportional to domain size
        freq_x = np.random.uniform(0, 10) / x_range if x_range > 0 else 0
        freq_y = np.random.uniform(0, 10) / y_range if y_range > 0 else 0
        freq_z = np.random.uniform(0, 10) / z_range if z_range > 0 else 0

        # Random phases
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        phase_z = np.random.uniform(0, 2 * np.pi)

        # Random amplitude
        amplitude = np.random.uniform(0.5, 1.5)

        # Create component: sin(freq_x * x + phase_x) * sin(freq_y * y + phase_y) * sin(freq_z * z + phase_z)
        component = amplitude * np.sin(freq_x * x + phase_x) * np.sin(freq_y * y + phase_y) * np.sin(freq_z * z + phase_z)

        # Add to function
        function_values += component

    # Normalize to have unit L2 norm
    function_values = function_values / np.linalg.norm(function_values)

    return function_values


def create_spatial_probe_function(points, k_power=None, seed=None):
    """Create a spatial probe function to capture high-frequency behaviors.

    Function defined as: f(p) = (1/2k) * sin(k * ψ * (ax + by + cz) + φ)
    where:
    - k is 2^(m/2) for m in {0,1,2,...,13}
    - ψ is frequency noise uniformly sampled in [0.75, 1.25]
    - φ is a random phase in [0, 2π]
    - a, b, c are randomly sampled with a+b+c=1

    Parameters:
    -----------
    points : array
        The point cloud coordinates of shape (n_points, 3).
    k_power : float or None
        If provided, uses 2^(k_power/2) as the k value. If None, randomly selects a power.
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    function_values : array
        Function values at each point, normalized to have unit L2 norm.
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Set k = 2^(m/2) for m randomly selected from {0,1,2,3,...,13}
    if k_power is None:
        m = np.random.randint(0, 14)  # m in {0,1,2,...,13}
    else:
        m = k_power
    k = 2 ** (m / 2)

    # Generate frequency noise ψ in [0.75, 1.25]
    psi = np.random.uniform(0.75, 1.25)

    # Generate random phase φ in [0, 2π]
    phi = np.random.uniform(0, 2 * np.pi)

    # Generate random weights a, b, c such that a+b+c=1
    # First generate random non-negative values
    a_raw = np.random.uniform(0, 1)
    b_raw = np.random.uniform(0, 1)
    c_raw = np.random.uniform(0, 1)
    # Normalize to make sum=1
    total = a_raw + b_raw + c_raw
    a = a_raw / total
    b = b_raw / total
    c = c_raw / total

    # Compute the function values
    projection = a * x + b * y + c * z
    function_values = (1 / (2 * k)) * np.sin(k * psi * projection + phi)

    # Normalize to have unit L2 norm
    function_values = function_values / np.linalg.norm(function_values)

    return function_values


def compute_reconstruction_error(function, basis, k):
    """
    Compute reconstruction error when approximating function with first k basis vectors.

    Parameters:
    -----------
    function : array
        The function values at each point.
    basis : array
        The basis vectors (each column is a basis vector).
    k : int
        Number of basis vectors to use.

    Returns:
    --------
    error : float
        L2 norm of the difference between the function and its projection.
    """
    # Get the first k basis vectors
    basis_k = basis[:, :k]

    # Compute the projection coefficients
    coeffs = basis_k.T @ function

    # Reconstruct the function
    reconstruction = basis_k @ coeffs

    # Compute the error
    error = np.linalg.norm(function - reconstruction)

    return error


def main():
    parser = argparse.ArgumentParser(description='Visualize Laplacian eigenvectors and solve generalized eigenvalue problem')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (.obj, .off, etc.)')
    parser.add_argument('--k', type=int, default=20, help='Number of eigenvectors to compute')
    parser.add_argument('--show_k', type=int, default=20, help='Number of eigenvectors to visualize')
    parser.add_argument('--max_basis_k', type=int, default=20, help='Maximum k for basis in generalized eigenvalue problem')
    parser.add_argument('--solver', type=str, default='torch_lobpcg',
                        choices=['eigsh', 'lobpcg', 'dense', 'torch_lobpcg'],
                        help='Solver for generalized eigenvalue problem')
    parser.add_argument('--n_rbf', type=int, default=10, help='Number of random RBF functions to generate')
    parser.add_argument('--rbf_centers', type=int, default=10, help='Number of centers for each RBF function')
    parser.add_argument('--rbf_type', type=str, default='thin_plate',
                        choices=['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate'],
                        help='Type of RBF function to use')
    parser.add_argument('--n_sin', type=int, default=10, help='Number of random sinusoidal functions to generate')
    parser.add_argument('--sin_components', type=int, default=5, help='Number of sinusoidal components per function')
    parser.add_argument('--n_probe', type=int, default=10, help='Number of spatial probe functions to generate')
    args = parser.parse_args()

    # Initialize polyscope
    ps.init()

    # Load the mesh using trimesh
    print(f"Loading mesh from {args.mesh_path}...")
    tm_mesh = trimesh.load(args.mesh_path)
    vertices = np.array(tm_mesh.vertices, dtype=np.float64)
    faces = np.array(tm_mesh.faces, dtype=np.int32)

    # Create pyFM mesh from vertices and faces
    mesh = TriMesh(vertices, faces)

    # Process the mesh and compute the regular Laplacian eigendecomposition
    print(f"Computing regular Laplacian eigendecomposition with k={args.k}...")
    mesh.process(k=args.k, intrinsic=False, verbose=True)

    # Retrieve eigenvalues and eigenvectors for the regular Laplacian
    regular_eigenvalues = mesh.eigenvalues
    regular_eigenvectors = mesh.eigenvectors

    # Get the cotangent weights (W) and vertex areas (A) from the mesh
    W = mesh.W  # Cotangent weights matrix (sparse)
    A = sparse.diags(mesh.vertex_areas)  # Vertex areas as diagonal matrix

    # Compute regular Laplacian L = A^(-1) * W
    # We need to convert W to CSC format for sparse solving
    L_regular = sparse.linalg.spsolve(A, W.tocsc())
    L_regular = sparse.csr_matrix(L_regular)  # Convert back to sparse format

    # Compute normalized Laplacian L_norm = A^(-1/2) * W * A^(-1/2)
    A_sqrt_inv = sparse.diags(1.0 / np.sqrt(mesh.vertex_areas))
    L_normalized = A_sqrt_inv @ W @ A_sqrt_inv

    # Make sure L_normalized is in CSR format for the eigenvalue solver
    L_normalized = L_normalized.tocsr()

    # Transform the eigenvectors to get the normalized Laplacian eigenvectors
    A_sqrt = sparse.diags(np.sqrt(mesh.vertex_areas))
    normalized_eigenvalues = regular_eigenvalues.copy()
    normalized_eigenvectors = A_sqrt @ regular_eigenvectors

    # Normalize the eigenvectors to have unit norm in the Euclidean sense
    for i in range(normalized_eigenvectors.shape[1]):
        normalized_eigenvectors[:, i] = normalized_eigenvectors[:, i] / np.linalg.norm(normalized_eigenvectors[:, i])

    # Register the point cloud with polyscope
    ps_cloud = ps.register_point_cloud("vertices", vertices)
    ps_cloud.set_color((0.8, 0.8, 0.8))

    # Visualize the first k eigenvectors for both Laplacian types
    show_k = min(args.show_k, args.k)

    # Visualize regular Laplacian eigenvectors
    for i in range(show_k):
        scalar_name = f"Regular Eigenvector {i:02d}"
        ps_cloud.add_scalar_quantity(scalar_name, regular_eigenvectors[:, i], enabled=i == 0, cmap='jet')

    # Visualize normalized Laplacian eigenvectors
    for i in range(show_k):
        scalar_name = f"Normalized Eigenvector {i:02d}"
        ps_cloud.add_scalar_quantity(scalar_name, normalized_eigenvectors[:, i], enabled=i == 0, cmap='jet')

    # Solve the generalized eigenvalue problem for different k values
    print("\nSolving generalized eigenvalue problem for different basis sizes:")
    max_basis_k = min(args.max_basis_k, args.k - 1)  # Ensure we have enough eigenvectors

    worst_case_eigenvectors = []
    worst_case_eigenvalues = []

    # Initialize random guess vector for LOBPCG - same for all k values
    if args.solver == 'lobpcg' or args.solver == 'torch_lobpcg':
        np.random.seed(42)  # For reproducibility
        X_guess = np.random.rand(vertices.shape[0], 1)
        X_guess = X_guess / np.linalg.norm(X_guess)

    # Use the normalized Laplacian for the generalized eigenvalue problem as in the notes
    # The problem is: (I - B_k B_k^T) f = λ L_norm f

    for basis_k in range(1, max_basis_k + 1):
        print(f"  Computing for basis size k={basis_k}...")

        # Create B_k using the normalized eigenvectors
        B_k = normalized_eigenvectors[:, :basis_k]

        # Compute I - B_k B_k^T
        B_k_B_k_T = B_k @ B_k.T
        I_minus_B_k_B_k_T = sparse.eye(vertices.shape[0]) - B_k_B_k_T

        # Convert to CSR format for the eigenvalue solver
        I_minus_B_k_B_k_T = sparse.csr_matrix(I_minus_B_k_B_k_T)

        # Solve generalized eigenvalue problem: (I - B_k B_k^T) f = λ L_norm f
        try:
            if args.solver == 'eigsh':
                # Use eigsh to find the largest eigenvalues
                # Note: For the generalized eigenvalue problem A*x = lambda*B*x
                # where A = I_minus_B_k_B_k_T and B = L_normalized
                eigenvalues, eigenvectors = splinalg.eigsh(
                    A=I_minus_B_k_B_k_T,  # The stiffness matrix
                    k=1,  # Number of eigenvalues to compute
                    M=L_normalized,  # The mass matrix for generalized eigenvalue problem
                    which='LM'  # Largest magnitude eigenvalues
                )
            elif args.solver == 'lobpcg':
                # Use the pre-initialized random guess vector for all k values
                # The generalized eigenvalue problem is: A*x = lambda*B*x
                # where A = I_minus_B_k_B_k_T and B = L_normalized
                eigenvalues, eigenvectors = splinalg.lobpcg(
                    A=I_minus_B_k_B_k_T,  # The stiffness matrix
                    X=X_guess,  # Initial guess for eigenvectors
                    B=L_normalized,  # The mass matrix for generalized eigenvalue problem
                    M=None,  # Preconditioner (None = no preconditioning)
                    Y=None,  # Constraints (None = no constraints)
                    largest=True,  # Find largest eigenvalues
                    maxiter=500  # Maximum number of iterations
                )
            elif args.solver == 'torch_lobpcg':
                # Convert sparse matrices to PyTorch tensors
                print("    Using PyTorch LOBPCG solver...")
                # Convert to dense if needed - PyTorch lobpcg works better with dense matrices for smaller problems
                A_dense = I_minus_B_k_B_k_T.toarray()
                B_dense = L_normalized.toarray()

                # Convert to PyTorch tensors
                A_torch = torch.tensor(A_dense, dtype=torch.float64)
                B_torch = torch.tensor(B_dense, dtype=torch.float64)

                # Create initial guess tensor
                X_torch = torch.tensor(X_guess, dtype=torch.float64)

                # Use PyTorch's LOBPCG
                E_torch, V_torch = torch.lobpcg(
                    A=A_torch,  # The stiffness matrix
                    k=1,  # Number of eigenvalues to compute
                    B=B_torch,  # The mass matrix
                    X=X_torch,  # Initial guess
                    largest=True,  # Find largest eigenvalues
                    method='ortho',  # More robust method
                    tol=1e-8,  # Tolerance
                    niter=500  # Maximum iterations
                )

                # Convert back to numpy
                eigenvalues = E_torch.cpu().numpy()
                eigenvectors = V_torch.cpu().numpy()
            elif args.solver == 'dense':
                # Convert to dense for small meshes
                if vertices.shape[0] < 5000:  # Only for reasonably sized meshes
                    A_dense = I_minus_B_k_B_k_T.toarray()
                    B_dense = L_normalized.toarray()
                    eigenvalues, eigenvectors = np.linalg.eig(A_dense, B_dense)
                    # Get the largest eigenvalue and its eigenvector
                    max_idx = np.argmax(eigenvalues)
                    eigenvalues = np.array([eigenvalues[max_idx]])
                    eigenvectors = eigenvectors[:, max_idx:max_idx + 1]
                else:
                    raise ValueError("Mesh too large for dense solver")
        except Exception as e:
            print(f"    Error with {args.solver} solver: {e}")
            print("    Trying alternative approach...")
            # Alternative approach: convert to dense for small meshes
            if vertices.shape[0] < 5000:  # Only for reasonably sized meshes
                A_dense = I_minus_B_k_B_k_T.toarray()
                B_dense = L_normalized.toarray()
                eigenvalues, eigenvectors = np.linalg.eig(A_dense, B_dense)
                # Get the largest eigenvalue and its eigenvector
                max_idx = np.argmax(eigenvalues)
                eigenvalues = np.array([eigenvalues[max_idx]])
                eigenvectors = eigenvectors[:, max_idx:max_idx + 1]
            else:
                print("    Mesh too large for dense solver. Skipping.")
                continue

        # Normalize the eigenvector to have unit norm
        eigenvectors[:, 0] = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])

        worst_case_eigenvalues.append(eigenvalues[0])
        worst_case_eigenvectors.append(eigenvectors[:, 0])

        # Visualize the worst-case function for this basis size
        scalar_name = f"Worst Case Function k={basis_k:02d}"
        ps_cloud.add_scalar_quantity(scalar_name, eigenvectors[:, 0], enabled=False, cmap='jet')

        print(f"    Eigenvalue λ = {eigenvalues[0]:.6f}")

    # Show projection coefficients of worst-case functions onto eigenfunctions
    print("\nProjection coefficients of worst-case functions onto eigenfunctions:")
    for basis_k in range(1, max_basis_k + 1):
        worst_case_idx = basis_k - 1
        worst_func = worst_case_eigenvectors[worst_case_idx]

        # Compute projection coefficients onto all eigenvectors
        coeffs = normalized_eigenvectors.T @ worst_func

        # Print the coefficients
        print(f"\n  Worst-case function for k={basis_k:02d} projections:")
        for i in range(min(args.k, 15)):  # Limit to first 15 coefficients for readability
            print(f"    Coefficient for eigenvector {i:02d}: {coeffs[i]:.8f}")

        # Print the L2 norm of coefficients outside the first k eigenvectors
        outside_norm = np.linalg.norm(coeffs[basis_k:])
        print(f"    L2 norm of coefficients outside first {basis_k} eigenvectors: {outside_norm:.8f}")

    # Print eigenvalues of worst-case functions
    print("\nEigenvalues of worst-case functions:")
    for i, eigen_val in enumerate(worst_case_eigenvalues):
        print(f"  k={i + 1:02d}: λ = {eigen_val:.6f}")

    # Generate random smooth functions using RBFs and compare reconstruction errors
    print("\nGenerating random smooth functions using RBFs...")
    rbf_functions = []
    for i in range(args.n_rbf):
        rbf_func = create_rbf_function(
            points=vertices,
            n_centers=args.rbf_centers,
            rbf_type=args.rbf_type,
            seed=42 + i
        )
        rbf_functions.append(rbf_func)

        # Visualize the RBF function
        scalar_name = f"RBF Function {i:02d}"
        ps_cloud.add_scalar_quantity(scalar_name, rbf_func, enabled=False, cmap='jet')

    # Generate random sinusoidal functions and compare reconstruction errors
    print("\nGenerating random sinusoidal functions...")
    sin_functions = []
    for i in range(args.n_sin):
        sin_func = create_sinusoidal_function(
            points=vertices,
            n_components=args.sin_components,
            seed=100 + i
        )
        sin_functions.append(sin_func)

        # Visualize the sinusoidal function
        scalar_name = f"Sinusoidal Function {i:02d}"
        ps_cloud.add_scalar_quantity(scalar_name, sin_func, enabled=False, cmap='jet')

    # Generate spatial probe functions to capture high-frequency behaviors
    print("\nGenerating spatial probe functions...")
    probe_functions = []
    # Use specific k values that increase in frequency
    # k = 2^(m/2) where m ranges from 0 to min(13, n_probe-1)
    max_m = min(13, args.n_probe - 1)
    for i in range(args.n_probe):
        # Create a range of frequencies, distributing them evenly
        if args.n_probe <= 14:
            # Use unique m values if we have 14 or fewer probe functions
            m = i % 14
        else:
            # Otherwise distribute them with emphasis on covering the range
            m = int((i / args.n_probe) * (max_m + 1))

        probe_func = create_spatial_probe_function(
            points=vertices,
            k_power=m,
            seed=200 + i
        )
        probe_functions.append(probe_func)

        # Visualize the probe function
        scalar_name = f"Probe Function k=2^({m / 2:.1f}) {i:02d}"
        ps_cloud.add_scalar_quantity(scalar_name, probe_func, enabled=False, cmap='jet')

    # Compare reconstruction errors between worst-case functions and RBF functions
    print("\nComparing reconstruction errors:")
    print("\nReconstruction errors for worst-case functions:")
    for basis_k in range(1, max_basis_k + 1):
        worst_case_idx = basis_k - 1
        worst_func = worst_case_eigenvectors[worst_case_idx]

        # Compute errors for different basis sizes
        errors = []
        for k in range(1, max_basis_k + 1):
            error = compute_reconstruction_error(worst_func, normalized_eigenvectors, k)
            errors.append(error)

        print(f"  Worst-case function for k={basis_k:02d}:")
        for k, error in enumerate(errors):
            print(f"    Using {k + 1:02d} basis vectors: error={error:.6f}")

    print("\nReconstruction errors for RBF functions:")
    for i, rbf_func in enumerate(rbf_functions):
        # Compute errors for different basis sizes
        errors = []
        for k in range(1, max_basis_k + 1):
            error = compute_reconstruction_error(rbf_func, normalized_eigenvectors, k)
            errors.append(error)

        print(f"  RBF function {i:02d}:")
        for k, error in enumerate(errors):
            print(f"    Using {k + 1:02d} basis vectors: error={error:.6f}")

    print("\nReconstruction errors for sinusoidal functions:")
    for i, sin_func in enumerate(sin_functions):
        # Compute errors for different basis sizes
        errors = []
        for k in range(1, max_basis_k + 1):
            error = compute_reconstruction_error(sin_func, normalized_eigenvectors, k)
            errors.append(error)

        print(f"  Sinusoidal function {i:02d}:")
        for k, error in enumerate(errors):
            print(f"    Using {k + 1:02d} basis vectors: error={error:.6f}")

    print("\nReconstruction errors for spatial probe functions:")
    for i, probe_func in enumerate(probe_functions):
        # Compute errors for different basis sizes
        errors = []
        for k in range(1, max_basis_k + 1):
            error = compute_reconstruction_error(probe_func, normalized_eigenvectors, k)
            errors.append(error)

        # Determine which k power was used (for clearer output)
        m = i % 14 if args.n_probe <= 14 else int((i / args.n_probe) * (min(13, args.n_probe - 1) + 1))

        print(f"  Probe function k=2^({m / 2:.1f}) {i:02d}:")
        for k, error in enumerate(errors):
            print(f"    Using {k + 1:02d} basis vectors: error={error:.6f}")

    # Show the polyscope GUI
    print("\nStarting polyscope GUI...")
    ps.show()


if __name__ == "__main__":
    main()