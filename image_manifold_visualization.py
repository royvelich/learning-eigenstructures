"""
Image Manifold Visualization Script

This script loads a trained LaplacianPredictorModuleImageManifold model from a checkpoint
and performs forward passes on the validation dataset. It's designed for exploring
predictions on image manifold data (e.g., CLIP/DINO embeddings).

Usage:
    python image_manifold_visualization.py \
        --config-name=your_config \
        globals.ckpt_path=/path/to/checkpoint.ckpt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Dict, Tuple, Optional
import pytorch_lightning as pl
from neural_laplacian.modules.laplacian_modules import LaplacianPredictorModuleImageManifold


def load_class_names(class_names_file: Optional[Path]) -> Optional[Dict[int, str]]:
    """
    Load class names from a text file.

    Args:
        class_names_file: Path to text file with class names (one per line, 0-indexed)

    Returns:
        Dictionary mapping class_id -> class_name, or None if file not provided/invalid
    """
    if class_names_file is None:
        return None

    class_names_path = Path(class_names_file)
    if not class_names_path.exists():
        print(f"Warning: Class names file not found: {class_names_file}")
        return None

    try:
        with open(class_names_path, 'r') as f:
            lines = f.readlines()

        # Create mapping: line index -> class name (strip whitespace)
        class_names = {i: line.strip() for i, line in enumerate(lines) if line.strip()}

        print(f"oe Loaded {len(class_names)} class names from {class_names_file}")
        return class_names

    except Exception as e:
        print(f"Warning: Failed to load class names from {class_names_file}: {e}")
        return None


def get_class_label(class_id: int, class_names: Optional[Dict[int, str]]) -> str:
    """
    Get class label (name if available, otherwise 'Class X').

    Args:
        class_id: Integer class ID
        class_names: Optional mapping of class_id -> class_name

    Returns:
        Class label string
    """
    if class_names is not None and class_id in class_names:
        return class_names[class_id]
    else:
        return f"Class {class_id}"


def load_model_from_checkpoint(ckpt_path: str, device: str = 'cpu') -> LaplacianPredictorModuleImageManifold:
    """
    Load trained model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from: {ckpt_path}")

    # Load the model
    model = LaplacianPredictorModuleImageManifold.load_from_checkpoint(
        ckpt_path,
        map_location=device
    )
    model.eval()  # Set to evaluation mode
    model.to(device)

    print("oe Model loaded successfully!")
    return model


def plot_eigenvector_embedding_2d(
    eigenvectors: torch.Tensor,
    class_ids: np.ndarray,
    item_idx: int,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Path = None
):
    """
    Plot 2D embedding using first two non-trivial eigenvectors.

    Args:
        eigenvectors: [N, K] eigenvector matrix
        class_ids: [N] numpy array of class labels
        item_idx: Index for title/filename
        class_names: Optional mapping of class_id -> class_name
        save_path: Optional directory to save plot
    """
    # Extract coordinates (skip first eigenvector which is constant)
    x = eigenvectors[:, 1].cpu().numpy()  # 2nd eigenvector (1st non-trivial)
    y = eigenvectors[:, 2].cpu().numpy()  # 3rd eigenvector (2nd non-trivial)

    # Determine number of classes for colormap selection
    n_classes = len(np.unique(class_ids))
    if n_classes <= 10:
        cmap = 'tab10'
    elif n_classes <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with class colors
    scatter = ax.scatter(x, y, c=class_ids, cmap=cmap,
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class ID', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel('Eigenvector 1 (2nd smallest eigenvalue)')
    ax.set_ylabel('Eigenvector 2 (3rd smallest eigenvalue)')
    ax.set_title(f'2D Eigenvector Embedding - Item {item_idx}')
    ax.grid(True, alpha=0.3)

    # Add legend with class counts
    unique_classes = np.unique(class_ids)
    handles = []
    labels = []
    for cls in unique_classes[:10]:  # Limit to 10 classes in legend to avoid clutter
        count = np.sum(class_ids == cls)
        # Create a dummy plot for legend
        handle = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=plt.cm.get_cmap(cmap)(cls / max(n_classes - 1, 1)),
                           markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        handles.append(handle)
        # Use class name if available
        class_label = get_class_label(cls, class_names)
        labels.append(f'{class_label} (n={count})')

    if len(unique_classes) > 10:
        labels.append(f'... and {len(unique_classes) - 10} more classes')

    ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f'eigenvector_2d_item_{item_idx:03d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved plot to: {filename}")

    plt.show()


def plot_single_embedding(
    ax,
    embedding: np.ndarray,
    class_ids: np.ndarray,
    title: str,
    class_names: Optional[Dict[int, str]] = None
):
    """
    Plot a single 2D embedding on the given axis.

    Args:
        ax: Matplotlib axis
        embedding: [N, 2] array of 2D coordinates
        class_ids: [N] array of class labels
        title: Title for this subplot
    """
    # Determine colormap based on number of classes
    n_classes = len(np.unique(class_ids))
    if n_classes <= 10:
        cmap = 'tab10'
    elif n_classes <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'

    # Scatter plot - removed edgecolors, increased alpha from 0.6 to 0.8
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=class_ids, cmap=cmap,
                        alpha=0.4, s=50)  # Removed edgecolors and linewidth

    # Increased font sizes
    ax.set_title(title, fontsize=18, fontweight='bold')  # Increased from 12 to 18
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Component 1', fontsize=18)  # Added fontsize=16
    ax.set_ylabel('Component 2', fontsize=18)  # Added fontsize=16

    # Increase tick label font size and tick thickness
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=8)  # Added width and length

    # Make frame (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    return scatter


def plot_comparison_embeddings_2d(
    eigenvectors: torch.Tensor,
    raw_features: np.ndarray,
    class_ids: np.ndarray,
    edge_index: torch.Tensor,
    item_idx: int,
    k_neighbors: int,
    use_cosine: bool,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Path = None
):
    """
    Plot comparison of learned eigenvectors vs classical manifold learning methods + graph Laplacian.

    Creates a 2x3 grid showing:
    - Learned eigenvectors (top-left)
    - Graph Laplacian (top-middle)
    - PCA (top-right)
    - UMAP (bottom-left)
    - Isomap (bottom-middle)
    - t-SNE (bottom-right)

    Args:
        eigenvectors: [N, K] predicted eigenvector matrix
        raw_features: [N, D] raw DINO/CLIP features
        class_ids: [N] numpy array of class labels
        edge_index: [2, E] edge connectivity from k-NN graph
        item_idx: Index for title/filename
        k_neighbors: Number of neighbors for manifold methods
        use_cosine: Whether to use cosine distance
        class_names: Optional mapping of class_id -> class_name
        save_path: Optional directory to save plot
    """
    from neural_laplacian import utils

    print(f"  Computing baseline embeddings (k_neighbors={k_neighbors})...")

    # Compute graph Laplacian eigenvectors
    pos_torch = torch.from_numpy(raw_features).float()
    # Ensure edge_index is on CPU for scipy operations
    edge_index_cpu = edge_index.cpu() if edge_index.is_cuda else edge_index
    graph_laplacian_eigenvectors = utils.compute_graph_laplacian_eigenvectors(
        edge_index=edge_index_cpu,
        pos=pos_torch,
        n_eigenvectors=3,  # Need at least 3 for 2D plot (skip first)
        use_cosine=use_cosine
    )
    graph_laplacian_embedding = graph_laplacian_eigenvectors[:, 1:3].cpu().numpy()

    # Compute baseline embeddings using utils
    embeddings_2d = utils.compute_manifold_embeddings(
        features=raw_features,
        n_components=2,
        k_neighbors=k_neighbors,
        methods=['pca', 'umap', 'tsne', 'isomap'],
        use_cosine=use_cosine,
        random_state=42
    )

    # Prepare learned eigenvectors (skip first, use next 2)
    learned_embedding = eigenvectors[:, 1:3].cpu().numpy()

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Determine colormap based on number of classes
    n_classes = len(np.unique(class_ids))
    if n_classes <= 10:
        cmap = 'tab10'
    elif n_classes <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'

    # Track which methods succeeded
    methods_plotted = []

    # Plot learned eigenvectors (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = plot_single_embedding(ax1, learned_embedding, class_ids,
                                    'Learned Eigenvectors', class_names)
    methods_plotted.append('Learned')

    # Plot Graph Laplacian (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_single_embedding(ax2, graph_laplacian_embedding, class_ids,
                         'Graph Laplacian', class_names)
    methods_plotted.append('Graph Laplacian')

    # Plot PCA (top-right)
    if embeddings_2d['pca'] is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        plot_single_embedding(ax3, embeddings_2d['pca'], class_ids, 'PCA', class_names)
        methods_plotted.append('PCA')

    # Plot UMAP (bottom-left)
    if embeddings_2d['umap'] is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        plot_single_embedding(ax4, embeddings_2d['umap'], class_ids, 'UMAP', class_names)
        methods_plotted.append('UMAP')

    # Plot Isomap (bottom-middle)
    if embeddings_2d['isomap'] is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        plot_single_embedding(ax5, embeddings_2d['isomap'], class_ids, 'Isomap', class_names)
        methods_plotted.append('Isomap')

    # Plot t-SNE (bottom-right)
    if embeddings_2d['tsne'] is not None:
        ax6 = fig.add_subplot(gs[1, 2])
        plot_single_embedding(ax6, embeddings_2d['tsne'], class_ids, 't-SNE', class_names)
        methods_plotted.append('t-SNE')

    # Add shared legend/colorbar
    # Create a legend in the remaining space or as a separate element
    unique_classes = np.unique(class_ids)
    if len(unique_classes) <= 10:
        # Add legend with class names
        handles = []
        labels_list = []
        for cls in unique_classes:
            count = np.sum(class_ids == cls)
            handle = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.get_cmap(cmap)(cls / max(n_classes - 1, 1)),
                               markersize=8, markeredgecolor='black', markeredgewidth=0.5)
            handles.append(handle)
            class_label = get_class_label(cls, class_names)
            labels_list.append(f'{class_label} (n={count})')

        # Add legend to the figure
        fig.legend(handles, labels_list, loc='center right', fontsize=13, framealpha=0.9,
                  bbox_to_anchor=(0.98, 0.5))
    else:
        # For many classes, add a colorbar instead
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=class_ids.min(),
                                                     vmax=class_ids.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=fig.get_axes(), fraction=0.02, pad=0.02)
        cbar.set_label('Class ID', rotation=270, labelpad=20)

    # Main title
    fig.suptitle(f'Embedding Comparison - Item {item_idx}\n'
                f'Methods: {", ".join(methods_plotted)}',
                fontsize=14, fontweight='bold')

    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)

    # Save if path provided
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f'comparison_2d_item_{item_idx:03d}.svg'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot to: {filename}")

    plt.show()


def plot_baseline_embeddings_2d(
    raw_features: np.ndarray,
    class_ids: np.ndarray,
    edge_index: torch.Tensor,
    item_idx: int,
    k_neighbors: int,
    use_cosine: bool,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Path = None
):
    """
    Plot only baseline manifold learning methods (no learned eigenvectors).

    Creates a 2x3 grid showing:
    - Graph Laplacian (top-left)
    - PCA (top-middle)
    - UMAP (top-right)
    - Isomap (bottom-left)
    - t-SNE (bottom-middle)

    Args:
        raw_features: [N, D] raw DINO/CLIP features
        class_ids: [N] numpy array of class labels
        edge_index: [2, E] edge connectivity from k-NN graph
        item_idx: Index for title/filename
        k_neighbors: Number of neighbors for manifold methods
        use_cosine: Whether to use cosine distance
        class_names: Optional mapping of class_id -> class_name
        save_path: Optional directory to save plot
    """
    from neural_laplacian import utils

    print(f"  Computing baseline embeddings (k_neighbors={k_neighbors})...")

    # Compute graph Laplacian eigenvectors
    pos_torch = torch.from_numpy(raw_features).float()
    # Ensure edge_index is on CPU for scipy operations
    edge_index_cpu = edge_index.cpu() if edge_index.is_cuda else edge_index
    graph_laplacian_eigenvectors = utils.compute_graph_laplacian_eigenvectors(
        edge_index=edge_index_cpu,
        pos=pos_torch,
        n_eigenvectors=3,  # Need at least 3 for 2D plot (skip first)
        use_cosine=use_cosine
    )
    graph_laplacian_embedding = graph_laplacian_eigenvectors[:, 1:3].cpu().numpy()

    # Compute baseline embeddings using utils
    embeddings_2d = utils.compute_manifold_embeddings(
        features=raw_features,
        n_components=2,
        k_neighbors=k_neighbors,
        methods=['pca', 'umap', 'tsne', 'isomap'],
        use_cosine=use_cosine,
        random_state=42
    )

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Determine colormap based on number of classes
    n_classes = len(np.unique(class_ids))
    if n_classes <= 10:
        cmap = 'tab10'
    elif n_classes <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'

    # Track which methods succeeded
    methods_plotted = []

    # Plot Graph Laplacian (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_single_embedding(ax1, graph_laplacian_embedding, class_ids,
                         'Graph Laplacian', class_names)
    methods_plotted.append('Graph Laplacian')

    # Plot PCA (top-middle)
    if embeddings_2d['pca'] is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        plot_single_embedding(ax2, embeddings_2d['pca'], class_ids, 'PCA', class_names)
        methods_plotted.append('PCA')

    # Plot UMAP (top-right)
    if embeddings_2d['umap'] is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        plot_single_embedding(ax3, embeddings_2d['umap'], class_ids, 'UMAP', class_names)
        methods_plotted.append('UMAP')

    # Plot Isomap (bottom-left)
    if embeddings_2d['isomap'] is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        plot_single_embedding(ax4, embeddings_2d['isomap'], class_ids, 'Isomap', class_names)
        methods_plotted.append('Isomap')

    # Plot t-SNE (bottom-middle)
    if embeddings_2d['tsne'] is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        plot_single_embedding(ax5, embeddings_2d['tsne'], class_ids, 't-SNE', class_names)
        methods_plotted.append('t-SNE')

    # Add shared legend/colorbar
    # Create a legend in the remaining space or as a separate element
    unique_classes = np.unique(class_ids)
    if len(unique_classes) <= 10:
        # Add legend with class names
        handles = []
        labels_list = []
        for cls in unique_classes:
            count = np.sum(class_ids == cls)
            handle = plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.get_cmap(cmap)(cls / max(n_classes - 1, 1)),
                               markersize=8, markeredgecolor='black', markeredgewidth=0.5)
            handles.append(handle)
            class_label = get_class_label(cls, class_names)
            labels_list.append(f'{class_label} (n={count})')

        # Add legend to the figure
        fig.legend(handles, labels_list, loc='center right', fontsize=10, framealpha=0.9,
                  bbox_to_anchor=(0.98, 0.5))
    else:
        # For many classes, add a colorbar instead
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=class_ids.min(),
                                                     vmax=class_ids.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=fig.get_axes(), fraction=0.02, pad=0.02)
        cbar.set_label('Class ID', rotation=270, labelpad=20)

    # Main title
    fig.suptitle(f'Baseline Methods Comparison - Item {item_idx}\n'
                f'Methods: {", ".join(methods_plotted)}',
                fontsize=14, fontweight='bold')

    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)

    # Save if path provided
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f'baselines_2d_item_{item_idx:03d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved baselines plot to: {filename}")

    plt.show()


def evaluate_embeddings(
    eigenvectors: torch.Tensor,
    raw_features: np.ndarray,
    class_ids: np.ndarray,
    edge_index: torch.Tensor,
    k_values: List[int],
    k_neighbors: int,
    use_cosine: bool
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Evaluate all embedding methods across different k values and return NMI/ARI scores.

    Args:
        eigenvectors: [N, K] predicted eigenvector matrix
        raw_features: [N, D] raw DINO/CLIP features
        class_ids: [N] numpy array of class labels
        edge_index: [2, E] edge connectivity from k-NN graph
        k_values: List of k values (embedding dimensions) to evaluate
        k_neighbors: Number of neighbors for manifold methods
        use_cosine: Whether to use cosine distance

    Returns:
        Dict mapping method name -> k value -> (nmi, ari, completeness, ami, homogeneity, v_measure, fmi) tuple
        Example: {'learned': {5: (0.6, 0.55), 10: (0.65, 0.60)}, ...}
    """
    from neural_laplacian import utils

    # Determine number of clusters from ground truth
    n_clusters = len(np.unique(class_ids))

    # Initialize results dictionary
    # Each method stores k -> (nmi, ari, completeness, ami, homogeneity, v_measure, fmi)
    results = {
        'learned': {},
        'graph_laplacian': {},
        'pca': {},
        'umap': {},
        'isomap': {},
        'tsne': {}
    }

    # Evaluate for each k value
    for k in k_values:
        # Skip if k is too large for eigenvectors
        if k >= eigenvectors.shape[1]:
            continue

        # Evaluate learned eigenvectors
        learned_embedding = eigenvectors[:, 1:k+1].cpu().numpy()
        predicted = utils.compute_kmeans_clustering(
            embeddings=learned_embedding,
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        nmi, ari, completeness, ami, homogeneity, v_measure, fmi = utils.compute_clustering_metrics(predicted, class_ids)
        results['learned'][k] = (nmi, ari, completeness, ami, homogeneity, v_measure, fmi)

        # Evaluate graph Laplacian
        try:
            pos_torch = torch.from_numpy(raw_features).float()
            # Ensure edge_index is on CPU for scipy operations
            edge_index_cpu = edge_index.cpu() if edge_index.is_cuda else edge_index
            graph_laplacian_eigenvectors = utils.compute_graph_laplacian_eigenvectors(
                edge_index=edge_index_cpu,
                pos=pos_torch,
                n_eigenvectors=k+1,  # Need k+1 to skip first
                use_cosine=use_cosine
            )
            graph_laplacian_embedding = graph_laplacian_eigenvectors[:, 1:k+1].cpu().numpy()
            predicted = utils.compute_kmeans_clustering(
                embeddings=graph_laplacian_embedding,
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            nmi, ari, completeness, ami, homogeneity, v_measure, fmi = utils.compute_clustering_metrics(predicted, class_ids)
            results['graph_laplacian'][k] = (nmi, ari, completeness, ami, homogeneity, v_measure, fmi)
        except Exception as e:
            print(f"Warning: Failed to evaluate graph_laplacian at k={k}: {e}")

        # Compute baseline embeddings with k components
        baseline_embeddings = utils.compute_manifold_embeddings(
            features=raw_features,
            n_components=k,
            k_neighbors=k_neighbors,
            methods=['pca', 'umap', 'tsne', 'isomap'],
            use_cosine=use_cosine,
            random_state=42
        )

        # Evaluate each baseline
        for method_name, embedding in baseline_embeddings.items():
            if embedding is not None:
                try:
                    predicted = utils.compute_kmeans_clustering(
                        embeddings=embedding,
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=10
                    )
                    nmi, ari, completeness, ami, homogeneity, v_measure, fmi = utils.compute_clustering_metrics(predicted, class_ids)
                    results[method_name][k] = (nmi, ari, completeness, ami, homogeneity, v_measure, fmi)
                except Exception as e:
                    print(f"Warning: Failed to evaluate {method_name} at k={k}: {e}")
            else:
                print(f"Warning: {method_name} failed to compute embedding at k={k}")

    return results


def run_evaluation_mode(
    model,
    data_loader,
    k_neighbors: int,
    k_values: List[int],
    device: str,
    output_path: Path,
    class_names: Optional[Dict[int, str]] = None
):
    """
    Run evaluation mode: compute metrics for all items, print results, save CSV.

    Args:
        model: Trained model
        data_loader: Validation data loader
        k_neighbors: Number of neighbors for baseline methods
        k_values: List of k values to evaluate
        device: Device to run on
        output_path: Path to save CSV results
        class_names: Optional mapping of class_id -> class_name
    """
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("RUNNING EVALUATION MODE")
    print("=" * 60)
    print(f"k_neighbors: {k_neighbors}")
    print(f"k_values: {k_values}")
    print(f"Device: {device}")
    print("Processing all batches automatically (no plots)...")
    print("=" * 60 + "\n")

    # Get use_cosine from model's knn_graph_config
    use_cosine = model._knn_graph_config.cosine if hasattr(model, '_knn_graph_config') else False
    print(f"Using cosine distance: {use_cosine}\n")

    # Accumulator for metrics: method -> k -> metric_name -> list of values
    all_metrics = {
        'learned': {k: {'nmi': [], 'ari': [], 'completeness': [], 'ami': [], 'homogeneity': [], 'v_measure': [], 'fmi': []} for k in k_values},
        'graph_laplacian': {k: {'nmi': [], 'ari': [], 'completeness': [], 'ami': [], 'homogeneity': [], 'v_measure': [], 'fmi': []} for k in k_values},
        'pca': {k: {'nmi': [], 'ari': [], 'completeness': [], 'ami': [], 'homogeneity': [], 'v_measure': [], 'fmi': []} for k in k_values},
        'umap': {k: {'nmi': [], 'ari': [], 'completeness': [], 'ami': [], 'homogeneity': [], 'v_measure': [], 'fmi': []} for k in k_values},
        'isomap': {k: {'nmi': [], 'ari': [], 'completeness': [], 'ami': [], 'homogeneity': [], 'v_measure': [], 'fmi': []} for k in k_values},
        'tsne': {k: {'nmi': [], 'ari': [], 'completeness': [], 'ami': [], 'homogeneity': [], 'v_measure': [], 'fmi': []} for k in k_values}
    }

    total_items = 0

    # Process all batches with progress bar
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
        # Move batch to device
        batch = batch.to(device)

        with torch.no_grad():
            # Forward pass
            try:
                laplacian_prediction, batch_with_predictions = model.predict_step(batch, batch_idx)
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                continue

        # Extract results
        data_list = batch_with_predictions.to_data_list()

        # Process each item in batch
        for idx, (data, eigenvectors) in enumerate(zip(data_list, laplacian_prediction.eigenvectors_list)):
            # Skip items without class labels
            if not hasattr(data, 'class_ids') or len(data.class_ids) == 0:
                continue

            # Get raw features, class labels, and edge_index
            raw_features = data.pos.cpu().numpy()
            class_ids = data.class_ids
            edge_index = data.edge_index

            # Evaluate all methods at all k values
            try:
                item_results = evaluate_embeddings(
                    eigenvectors=eigenvectors,
                    raw_features=raw_features,
                    class_ids=class_ids,
                    edge_index=edge_index,
                    k_values=k_values,
                    k_neighbors=k_neighbors,
                    use_cosine=use_cosine
                )

                # Accumulate results
                for method, k_dict in item_results.items():
                    for k, (nmi, ari, completeness, ami, homogeneity, v_measure, fmi) in k_dict.items():
                        all_metrics[method][k]['nmi'].append(nmi)
                        all_metrics[method][k]['ari'].append(ari)
                        all_metrics[method][k]['completeness'].append(completeness)
                        all_metrics[method][k]['ami'].append(ami)
                        all_metrics[method][k]['homogeneity'].append(homogeneity)
                        all_metrics[method][k]['v_measure'].append(v_measure)
                        all_metrics[method][k]['fmi'].append(fmi)

                total_items += 1

            except Exception as e:
                print(f"\nError evaluating item {idx} in batch {batch_idx}: {e}")
                continue

    print(f"\n{'=' * 60}")
    print(f"Processed {total_items} items with class labels")
    print(f"{'=' * 60}\n")

    # Compute statistics and display/save results
    print_and_save_results(all_metrics, k_values, total_items, output_path, class_names)


def print_and_save_results(
    all_metrics: Dict[str, Dict[int, Dict[str, List[float]]]],
    k_values: List[int],
    total_items: int,
    output_path: Path,
    class_names: Optional[Dict[int, str]] = None
):
    """
    Print results to console and save to CSV.

    Args:
        all_metrics: Accumulated metrics
        k_values: List of k values evaluated
        total_items: Total number of items processed
        output_path: Path to save CSV
        class_names: Optional mapping of class_id -> class_name
    """
    import csv

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total items evaluated: {total_items}\n")

    # Prepare results for display and CSV
    csv_rows = []

    # For each method, compute mean and std across all k values
    for method in ['learned', 'graph_laplacian', 'pca', 'umap', 'isomap', 'tsne']:
        for k in k_values:
            nmi_list = all_metrics[method][k]['nmi']
            ari_list = all_metrics[method][k]['ari']
            completeness_list = all_metrics[method][k]['completeness']
            ami_list = all_metrics[method][k]['ami']
            homogeneity_list = all_metrics[method][k]['homogeneity']
            v_measure_list = all_metrics[method][k]['v_measure']
            fmi_list = all_metrics[method][k]['fmi']

            if len(nmi_list) > 0:
                nmi_mean = np.mean(nmi_list)
                nmi_std = np.std(nmi_list)
                ari_mean = np.mean(ari_list)
                ari_std = np.std(ari_list)
                completeness_mean = np.mean(completeness_list)
                completeness_std = np.std(completeness_list)
                ami_mean = np.mean(ami_list)
                ami_std = np.std(ami_list)
                homogeneity_mean = np.mean(homogeneity_list)
                homogeneity_std = np.std(homogeneity_list)
                v_measure_mean = np.mean(v_measure_list)
                v_measure_std = np.std(v_measure_list)
                fmi_mean = np.mean(fmi_list)
                fmi_std = np.std(fmi_list)
                n_items = len(nmi_list)

                csv_rows.append({
                    'method': method,
                    'k': k,
                    'nmi_mean': nmi_mean,
                    'nmi_std': nmi_std,
                    'ari_mean': ari_mean,
                    'ari_std': ari_std,
                    'completeness_mean': completeness_mean,
                    'completeness_std': completeness_std,
                    'ami_mean': ami_mean,
                    'ami_std': ami_std,
                    'homogeneity_mean': homogeneity_mean,
                    'homogeneity_std': homogeneity_std,
                    'v_measure_mean': v_measure_mean,
                    'v_measure_std': v_measure_std,
                    'fmi_mean': fmi_mean,
                    'fmi_std': fmi_std,
                    'n_items': n_items
                })

    # Print results in a nice table format
    print(f"{'Method':<10} | {'k':<3} | {'NMI (mean+/-std)':<20} | {'ARI (mean+/-std)':<20} | {'n_items':<8}")
    print("-" * 80)

    for row in csv_rows:
        print(f"{row['method']:<10} | {row['k']:<3} | "
              f"{row['nmi_mean']:.4f} +/- {row['nmi_std']:.4f}    | "
              f"{row['ari_mean']:.4f} +/- {row['ari_std']:.4f}    | "
              f"{row['n_items']:<8}")

    print("=" * 80 + "\n")

    # Find best results for each k
    print("Best Results per k:")
    print("-" * 80)
    for k in k_values:
        k_results = [r for r in csv_rows if r['k'] == k]
        if k_results:
            best_nmi = max(k_results, key=lambda x: x['nmi_mean'])
            best_ari = max(k_results, key=lambda x: x['ari_mean'])
            print(f"k={k:2d}: Best NMI = {best_nmi['method']:8s} ({best_nmi['nmi_mean']:.4f}), "
                  f"Best ARI = {best_ari['method']:8s} ({best_ari['ari_mean']:.4f})")
    print("=" * 80 + "\n")

    # Save to CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['method', 'k', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std',
                      'completeness_mean', 'completeness_std', 'ami_mean', 'ami_std',
                      'homogeneity_mean', 'homogeneity_std', 'v_measure_mean', 'v_measure_std',
                      'fmi_mean', 'fmi_std', 'n_items']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"✓ Results saved to: {output_path}")
    print("=" * 80 + "\n")


def process_batch(model, batch, batch_idx, device='cpu'):
    """
    Process a single batch through the model.

    Args:
        model: Trained LaplacianPredictorModuleImageManifold
        batch: Data batch
        batch_idx: Index of current batch
        device: Device to run inference on

    Returns:
        Tuple of (laplacian_prediction, batch_with_predictions, plot_data_list)
        where plot_data_list contains (eigenvectors, raw_features, class_ids, edge_index, item_idx, use_cosine) tuples
    """
    # Move batch to device
    batch = batch.to(device)

    with torch.no_grad():
        # Forward pass through the model
        laplacian_prediction, batch_with_predictions = model.predict_step(batch, batch_idx)

    # Extract results
    data_list = batch_with_predictions.to_data_list()

    # Get use_cosine from model's knn_graph_config
    use_cosine = model._knn_graph_config.cosine if hasattr(model, '_knn_graph_config') else False

    # Collect data for potential plotting
    plot_data_list = []

    # Print summary for each item in batch
    for idx, (data, eigenvectors, weights) in enumerate(
        zip(data_list,
            laplacian_prediction.eigenvectors_list,
            laplacian_prediction.weights_list)
    ):
        item_idx = batch_idx * len(data_list) + idx

        print(f"\n--- Item {item_idx} ---")
        print(f"  Images in manifold: {data.pos.shape[0]}")
        print(f"  Feature dimension: {data.pos.shape[1]}")
        print(f"  Predicted eigenvectors: {eigenvectors.shape}")
        print(f"  Predicted weights: {weights.shape}")

        # If we have eigenvalues from the loss
        if laplacian_prediction.weighted_eigenvalues_list is not None:
            eigenvalues = laplacian_prediction.weighted_eigenvalues_list[idx]
            print(f"  Weighted eigenvalues shape: {eigenvalues.shape}")
            print(f"  Eigenvalue range: [{eigenvalues.min().item():.6f}, {eigenvalues.max().item():.6f}]")

        # If class labels exist, show distribution and prepare plot data
        if hasattr(data, 'class_ids') and len(data.class_ids) > 0:
            # class_ids is a numpy array
            unique_classes = np.unique(data.class_ids)
            class_counts = np.bincount(data.class_ids)
            print(f"  Number of classes: {len(unique_classes)}")
            print(f"  Class distribution: {class_counts.tolist()}")

            # Check if we have enough eigenvectors for 2D plot
            if eigenvectors.shape[1] >= 3:
                # Store eigenvectors, raw features, class_ids, edge_index, item_idx, and use_cosine
                raw_features = data.pos.cpu().numpy()
                edge_index = data.edge_index
                plot_data_list.append((eigenvectors, raw_features, data.class_ids, edge_index, item_idx, use_cosine))
                print(f"  [Press 'p' for learned embedding, 'b' for baselines, 'a' for all]")
            else:
                print(f"  Not enough eigenvectors for 2D plot (need >= 3, have {eigenvectors.shape[1]})")

        # Show edge information if available
        if hasattr(data, 'edge_index'):
            print(f"  Graph edges: {data.edge_index.shape[1]}")

    return laplacian_prediction, batch_with_predictions, plot_data_list


def print_model_info(model):
    """Print information about the loaded model."""
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print spectral clustering k values if available
    if hasattr(model, '_spectral_clustering_k_values'):
        print(f"Spectral clustering k values: {model._spectral_clustering_k_values}")

    # Print manifold learning methods if available
    if hasattr(model, '_manifold_learning_methods'):
        print(f"Manifold learning methods: {model._manifold_learning_methods}")

    print("=" * 60)


@hydra.main(version_base="1.2", config_path="config/training")
def main(cfg: DictConfig):
    """Main function for image manifold visualization."""

    # Check for checkpoint path
    if not hasattr(cfg.globals, 'ckpt_path') or cfg.globals.ckpt_path is None:
        raise ValueError(
            "Please provide checkpoint path via cfg.globals.ckpt_path\n"
            "Example: python image_manifold_visualization.py globals.ckpt_path=/path/to/checkpoint.ckpt"
        )

    ckpt_path = cfg.globals.ckpt_path

    # Check if checkpoint exists
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Seed for reproducibility
    seed = cfg.globals.seed if hasattr(cfg.globals, 'seed') else 42
    pl.seed_everything(seed)
    print(f"Random seed set to: {seed}")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model from checkpoint
    model = load_model_from_checkpoint(ckpt_path, device=device)

    # Print model information
    print_model_info(model)

    # Instantiate data module
    print("\nInstantiating data module...")
    data_module = hydra.utils.instantiate(cfg.data_module.module)
    print("oe Data module created!")

    # Use validation data loader
    print("\nGetting validation data loader...")
    data_loader = data_module.val_dataloader()

    # Handle both single dataloader and list of dataloaders
    if isinstance(data_loader, list):
        print(f"Found {len(data_loader)} validation dataloaders")
        # Use the first one
        data_loader = data_loader[0]
        print("Using the first dataloader")

    # Get dataset info
    if hasattr(data_loader, 'dataset'):
        if hasattr(data_loader.dataset, 'name'):
            print(f"Dataset name: {data_loader.dataset.name}")
        if hasattr(data_loader.dataset, '__len__'):
            print(f"Dataset size: {len(data_loader.dataset)} items")

    print(f"Number of batches: {len(data_loader)}")

    # Get max batches from config if specified
    max_batches = cfg.get('max_batches', None)
    if max_batches is not None:
        print(f"Will process maximum {max_batches} batches")

    # Load class names if provided
    class_names_file = cfg.get('class_names_file', None)
    class_names = load_class_names(class_names_file)

    # Check if evaluation mode is enabled
    evaluation_mode = cfg.get('evaluation_mode', False)

    if evaluation_mode:
        # EVALUATION MODE: compute metrics, no interactive visualization
        print("\n" + "=" * 60)
        print("EVALUATION MODE ENABLED")
        print("=" * 60)

        # Get k_neighbors from config (for baseline methods)
        k_neighbors = cfg.get('k_neighbors', 30)
        print(f"k_neighbors: {k_neighbors}")

        # Get k_values for evaluation
        # Try to get from config, otherwise use model's spectral_clustering_k_values
        if 'evaluation_k_values' in cfg:
            k_values = list(cfg.evaluation_k_values)
        elif hasattr(model, '_spectral_clustering_k_values'):
            k_values = model._spectral_clustering_k_values
            print(f"Using model's spectral_clustering_k_values: {k_values}")
        else:
            k_values = [5, 10, 20]  # Default
            print(f"Using default k_values: {k_values}")

        # Output path for CSV - include checkpoint name
        # Extract checkpoint filename without extension
        ckpt_name = Path(ckpt_path).stem  # e.g., 'model_epoch10' from 'model_epoch10.ckpt'
        default_output = f'evaluation_results_{ckpt_name}.csv'
        output_path = Path(cfg.get('evaluation_output', default_output))

        # Run evaluation
        run_evaluation_mode(
            model=model,
            data_loader=data_loader,
            k_neighbors=k_neighbors,
            k_values=k_values,
            device=device,
            output_path=output_path,
            class_names=class_names
        )

        return  # Exit after evaluation

    # INTERACTIVE VISUALIZATION MODE (existing code)
    print("\n" + "=" * 60)
    print("STARTING VISUALIZATION LOOP")
    print("=" * 60)
    print("\nInteractive commands:")
    print("  [Enter]  - Continue to next batch")
    print("  'p'      - Plot learned eigenvector embeddings")
    print("  'b'      - Plot baseline methods (PCA, UMAP, Isomap, t-SNE)")
    print("  'a'      - Plot all (learned + baselines comparison)")
    print("  'q'      - Quit")

    # Get k_neighbors from config (for baseline methods)
    k_neighbors = cfg.get('k_neighbors', 30)  # Default to 30 if not specified
    print(f"\nUsing k_neighbors={k_neighbors} for baseline manifold learning methods")
    print("(Set with: +k_neighbors=N)")

    # Output directory for saving plots
    output_dir = Path("visualization_output")

    # Visualization loop
    batches_processed = 0
    for batch_idx, batch in enumerate(data_loader):
        print(f"\n{'=' * 60}")
        print(f"Processing Batch {batch_idx + 1}/{len(data_loader)}")
        print(f"{'=' * 60}")

        # Process batch
        try:
            laplacian_prediction, batch_with_predictions, plot_data_list = process_batch(
                model, batch, batch_idx, device=device
            )
            batches_processed += 1

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()

            user_input = input(f"\nError occurred. Press Enter to continue, 'q' to quit: ")
            if user_input.lower() == 'q':
                print(f"\nVisualization stopped due to error.")
                return
            continue

        # Check if we've reached max batches
        if max_batches is not None and batches_processed >= max_batches:
            print(f"\nReached maximum batch limit ({max_batches})")
            break

        # User interaction
        while True:
            user_input = input(f"\nBatch {batch_idx + 1} complete. [Enter] continue, 'p' learned, 'b' baselines, 'a' all, 'q' quit: ")

            if user_input.lower() == 'q':
                print(f"\nVisualization complete!")
                print(f"Processed {batches_processed} batches.")
                return

            elif user_input.lower() == 'p':
                # Plot learned eigenvectors only
                if len(plot_data_list) == 0:
                    print("No items with class labels in this batch to plot.")
                else:
                    print(f"\nPlotting learned eigenvectors for {len(plot_data_list)} items...")
                    for eigenvectors, raw_features, class_ids, edge_index, item_idx, use_cosine in plot_data_list:
                        plot_eigenvector_embedding_2d(
                            eigenvectors=eigenvectors,
                            class_ids=class_ids,
                            item_idx=item_idx,
                            class_names=class_names,
                            save_path=output_dir
                        )

            elif user_input.lower() == 'b':
                # Plot baseline methods only
                if len(plot_data_list) == 0:
                    print("No items with class labels in this batch to plot.")
                else:
                    print(f"\nPlotting baseline methods for {len(plot_data_list)} items...")
                    for eigenvectors, raw_features, class_ids, edge_index, item_idx, use_cosine in plot_data_list:
                        plot_baseline_embeddings_2d(
                            raw_features=raw_features,
                            class_ids=class_ids,
                            edge_index=edge_index,
                            item_idx=item_idx,
                            k_neighbors=k_neighbors,
                            use_cosine=use_cosine,
                            class_names=class_names,
                            save_path=output_dir
                        )

            elif user_input.lower() == 'a':
                # Plot all (comparison view)
                if len(plot_data_list) == 0:
                    print("No items with class labels in this batch to plot.")
                else:
                    print(f"\nPlotting comparison (learned + baselines) for {len(plot_data_list)} items...")
                    for eigenvectors, raw_features, class_ids, edge_index, item_idx, use_cosine in plot_data_list:
                        plot_comparison_embeddings_2d(
                            eigenvectors=eigenvectors,
                            raw_features=raw_features,
                            class_ids=class_ids,
                            edge_index=edge_index,
                            item_idx=item_idx,
                            k_neighbors=k_neighbors,
                            use_cosine=use_cosine,
                            class_names=class_names,
                            save_path=output_dir
                        )

            else:
                # Enter or any other key - continue to next batch
                break

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Processed all {batches_processed} batches!")


if __name__ == "__main__":
    main()