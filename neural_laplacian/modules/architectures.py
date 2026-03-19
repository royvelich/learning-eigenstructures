# standard library
from typing import List, Type, Callable, Optional, Dict

# neural laplacian
from neural_laplacian import utils

# torch
import torch
import torch.nn as nn

# torch_geometric
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn_interpolate, knn_graph

# torch_cluster
from torch_cluster import fps


class ConfigurableGNNBase(nn.Module):
    def __init__(self, conv_layers: List[torch_geometric.nn.conv.MessagePassing], mlp_layers: Optional[List[torch.nn.Module]], k: int, concat_residual: bool, recompute_knn: bool):
        super().__init__()
        self._conv_layers = torch.nn.ModuleList(conv_layers)
        if mlp_layers is not None:
            self._mlp_layers = torch.nn.ModuleList(mlp_layers)
        else:
            self._mlp_layers = [None for _ in range(len(self._conv_layers))]
        self._k = k
        self._recompute_knn = recompute_knn
        self._concat_residual = concat_residual
        self._conv_input_args = utils.get_input_args(forward_method=self._conv_layers[0].forward)


class ConfigurableGNN(ConfigurableGNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batch: Batch) -> torch.Tensor:
        # if not self._recompute_knn:
        #     batch['edge_index'] = knn_graph(x=batch['pos'], k=self._k, batch=batch['batch'])

        x_list = [batch['x']]
        for i, (conv_layer, mlp_layer) in enumerate(zip(self._conv_layers, self._mlp_layers)):
            # Prepare input arguments for the layer
            layer_inputs = {}
            for key in self._conv_input_args:
                if key == 'x':
                    layer_inputs[key] = x_list[-1]
                elif key == 'edge_index' and self._recompute_knn:
                    layer_inputs[key] = knn_graph(x=x_list[-1], k=self._k, batch=batch['batch'])
                else:
                    layer_inputs[key] = batch[key]

            # Apply the layer
            x = conv_layer(**layer_inputs)
            if mlp_layer is not None:
                x = mlp_layer(x)
            x_list.append(x)

        if self._concat_residual:
            return torch.concat(x_list[1:], dim=-1)

        return x_list[-1]


# https://discuss.pytorch.org/t/what-is-the-output-of-a-topkpooling-layer/125564/2
class ConfigurableUNetGNN(nn.Module):
    def __init__(self, encoders: List[nn.Module], decoders: List[nn.Module], pool_ratio: float, concat_residual: bool, k: int, **kwargs):
        super().__init__(**kwargs)
        self._pool_ratio = pool_ratio
        self._concat_residual = concat_residual
        self._k = k
        self._encoders = torch.nn.ModuleList(encoders)
        self._decoders = torch.nn.ModuleList(decoders)
        pass

    def forward(self, batch: Batch) -> torch.Tensor:
        def duplicate_batch(batch: Batch, idx: Optional[torch.Tensor] = None) -> Batch:
            all_args = ['pos', 'x']
            data_list = batch.to_data_list()
            new_data_list = []
            for data in data_list:
                new_data = Data()
                for arg in all_args:
                    current_data = data[arg]
                    if idx is not None:
                        current_data = current_data[idx]

                    new_data[arg] = current_data
                new_data_list.append(new_data)

            batch = Batch.from_data_list(new_data_list)
            return batch


        residual_batches = []
        batch = duplicate_batch(batch=batch)
        for i, encoder in enumerate(self._encoders):
            # First convert to individual graphs using the original batch
            data_list = batch.to_data_list()

            # Recalculate edges for each graph individually
            for data in data_list:
                data.edge_index = knn_graph(
                    x=data.pos,
                    k=self._k,
                    loop=False
                )

            # Create a fresh batch object
            batch = Batch.from_data_list(data_list)
            x = encoder(batch)
            batch = utils.rebuild_batch_from_tensor(batch=batch, property_name='x', property_tensor=x)
            residual_batches = [batch.clone()] + residual_batches
            if i < len(self._encoders) - 1:
                idx = fps(src=batch.pos, batch=batch.batch, ratio=self._pool_ratio)
                batch = duplicate_batch(batch=batch, idx=idx)

        for i, decoder in enumerate(self._decoders):
            if i == 0:
                # For first decoder iteration, start with the bottleneck features
                # from residual_batches[0], not the raw downsampled batch
                batch = residual_batches[0]

            # Get the target resolution level for upsampling
            target_batch = residual_batches[i + 1]

            # Interpolate current batch features to target batch resolution
            interpolated_x = knn_interpolate(
                x=batch.x,  # Current features (lower resolution)
                pos_x=batch.pos,  # Current positions (lower resolution)
                pos_y=target_batch.pos,  # Target positions (higher resolution)
                batch_x=batch.batch,  # Current batch indices
                batch_y=target_batch.batch,  # Target batch indices
                k=self._k
            )

            # Handle skip connection
            if self._concat_residual:
                combined_x = torch.cat([interpolated_x, target_batch.x], dim=1)
            else:
                combined_x = interpolated_x + target_batch.x

            # Update batch with combined features at target resolution
            batch = utils.rebuild_batch_from_tensor(
                batch=target_batch,
                property_name='x',
                property_tensor=combined_x
            )

            # Recalculate edges for the upsampled resolution
            data_list = batch.to_data_list()
            for data in data_list:
                data.edge_index = knn_graph(
                    x=data.pos,
                    k=self._k,
                    loop=False
                )
            batch = Batch.from_data_list(data_list)

            # Apply decoder at this resolution
            x = decoder(batch)
            batch = utils.rebuild_batch_from_tensor(
                batch=batch,
                property_name='x',
                property_tensor=x
            )

        return batch.x




class ConfigurableMLP(nn.Module):
    def __init__(self, channels: List[int], use_batch_norm: bool = False, use_layer_norm: bool = False, activation: Optional[str] = None):
        super().__init__()
        self._layers = nn.ModuleList()
        self._normalizations = nn.ModuleList()
        self._channels = channels
        self._use_batch_norm = use_batch_norm
        self._use_layer_norm = use_layer_norm
        
        if activation is not None and len(channels) > 2:
            self._activations = [utils.import_object(full_type_name=activation)() for _ in range(len(channels) - 1)]
        else:
            self._activations = None
            
        for i in range(len(channels) - 1):
            self._layers.append(nn.Linear(in_features=channels[i], out_features=channels[i+1]))
            
            # Add normalization layer if needed (except for the last layer)
            if i < len(channels) - 2:
                if self._use_batch_norm:
                    self._normalizations.append(nn.BatchNorm1d(num_features=channels[i+1]))
                elif self._use_layer_norm:
                    self._normalizations.append(nn.LayerNorm(normalized_shape=channels[i+1]))
                else:
                    self._normalizations.append(None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._layers[:-1]):
            x = layer(x)
            
            # Apply normalization if available
            if (self._use_batch_norm or self._use_layer_norm) and i < len(self._normalizations):
                if self._normalizations[i] is not None:
                    x = self._normalizations[i](x)
                    
            # Apply activation if available
            if self._activations is not None:
                x = self._activations[i](x)
                
        x = self._layers[-1](x)
        return x


class ConfigurablePooling(torch.nn.Module):
    def __init__(self, pooling_layers: List[str]):
        super().__init__()
        self._pooling_layers = pooling_layers

    def forward(self, batch: Batch, x: torch.Tensor) -> torch.Tensor:
        pooling_features_list = []
        for pooling_layer in self._pooling_layers:
            pooling_fn = utils.import_object(full_type_name=pooling_layer)
            pooling_out = pooling_fn(x=x, batch=batch.batch)
            pooling_features_list.append(pooling_out)

        pooling_cat = torch.cat(pooling_features_list, dim=1)

        return pooling_cat


class ConfigurableTransformer(nn.Module):
    """
    Configurable transformer for processing geometric data with global attention.

    This class adapts PyTorch's standard transformer encoder to work with geometric point clouds
    by treating each point cloud as a sequence where vertices are tokens. Uses global attention
    where each vertex can attend to all other vertices in the same point cloud.
    """

    def __init__(self,
                 transformer_encoder_layer: nn.TransformerEncoderLayer,
                 num_layers: int):
        """
        Initialize the configurable transformer base.

        Args:
            transformer_encoder_layer: Single transformer encoder layer to be replicated
            num_layers: Number of transformer layers to stack
        """
        super().__init__()

        # Extract d_model from the transformer layer
        self.d_model = transformer_encoder_layer.self_attn.embed_dim

        # Create the transformer encoder with batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_layers
        )

    def _prepare_sequences(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert batch data to padded sequences suitable for transformer processing.

        Args:
            batch: Batch containing point clouds with features in batch.x

        Returns:
            sequences: Padded sequences [batch_size, max_seq_len, feature_dim]
            padding_mask: Mask indicating padding positions [batch_size, max_seq_len]
        """
        data_list = batch.to_data_list()
        batch_size = len(data_list)

        # Find maximum sequence length in the batch
        seq_lengths = [data.x.shape[0] for data in data_list]
        max_seq_len = max(seq_lengths)
        feature_dim = data_list[0].x.shape[1]

        # Initialize padded sequences and mask
        sequences = torch.zeros(batch_size, max_seq_len, feature_dim,
                                device=batch.x.device, dtype=batch.x.dtype)
        padding_mask = torch.ones(batch_size, max_seq_len,
                                  device=batch.x.device, dtype=torch.bool)

        # Fill sequences and create padding mask
        start_idx = 0
        for i, data in enumerate(data_list):
            seq_len = data.x.shape[0]
            end_idx = start_idx + seq_len

            # Copy features to padded sequence
            sequences[i, :seq_len, :] = batch.x[start_idx:end_idx]

            # Mark non-padding positions as False (padding positions remain True)
            padding_mask[i, :seq_len] = False

            start_idx = end_idx

        return sequences, padding_mask

    def _unpad_and_flatten(self, output: torch.Tensor, batch: Batch) -> torch.Tensor:
        """
        Convert padded transformer output back to flat tensor format.

        Args:
            output: Transformer output [batch_size, max_seq_len, feature_dim]
            batch: Original batch for extracting sequence lengths

        Returns:
            Flattened output [total_num_vertices, feature_dim]
        """
        data_list = batch.to_data_list()
        output_list = []

        for i, data in enumerate(data_list):
            seq_len = data.x.shape[0]
            # Extract only the non-padded portion
            output_list.append(output[i, :seq_len, :])

        # Concatenate all sequences back to flat format
        return torch.cat(output_list, dim=0)

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            batch: Batch containing point clouds with features in batch.x

        Returns:
            Transformed features [total_num_vertices, d_model]
        """
        # Step 1: Prepare padded sequences for transformer
        sequences, padding_mask = self._prepare_sequences(batch)

        # Step 2: Apply transformer encoder
        # With batch_first=True, transformer expects [batch_size, seq_len, feature_dim]
        transformer_output = self.transformer_encoder(
            src=sequences,
            src_key_padding_mask=padding_mask
        )

        # Step 3: Unpad and flatten back to original format
        final_output = self._unpad_and_flatten(transformer_output, batch)

        return final_output