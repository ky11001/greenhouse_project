"""
Geometry Encoder for MT3 Hierarchical Retrieval

This module provides the trained geometry encoder used for demo retrieval in MT3.
It contains only inference code - training has been removed.

For MT3 deployment, this encoder:
1. Takes a segmented point cloud of an object
2. Encodes it into a 512-dimensional feature vector
3. Used to find visually similar demonstrations via cosine similarity
"""

import json
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import PointNetConv, fps, radius, MLP
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Union, Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops
import lightning as pl


# ============================================================================
# Core Encoder Architecture (Inference Only)
# ============================================================================

class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for 3D points."""

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False, add_original_x: bool = True):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space

        if add_original_x:
            self.embed_fns = [lambda x: x]
            self.d_output = d_input * (1 + 2 * self.n_freqs)
        else:
            self.embed_fns = []
            self.d_output = d_input * (2 * self.n_freqs)

        # Define frequencies
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        """Apply positional encoding to input."""
        return torch.cat([f(x) for f in self.embed_fns], dim=-1)


class PointNetConvPE(PointNetConv):
    """PointNet convolution with positional encoding."""

    def __init__(self, nn_dims, global_nn_dims=None, add_self_loops=False, aggr='mean',
                 num_freqs=4, cat_pos=False, radius=1., use_film=False, cond_dim=512):
        self.radius = radius
        nn_dims[0] += 3 * (2 * num_freqs)
        neur_net = MLP(nn_dims, norm=None, act=torch.nn.GELU(approximate='tanh'))
        if cat_pos and global_nn_dims is not None:
            global_nn_dims[0] += 3 * (2 * num_freqs + 1)

        global_nn = None if global_nn_dims is None else \
            MLP(global_nn_dims, norm=None, act=torch.nn.GELU(approximate='tanh'), plain_last=True)
        self.cat_pos = cat_pos
        super().__init__(neur_net, global_nn=global_nn, add_self_loops=add_self_loops, aggr=aggr)
        self.pe = PositionalEncoder(3, num_freqs)

        self.use_film = use_film
        if use_film:
            self.film = nn.Linear(cond_dim, global_nn_dims[0]*2)

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = self.pe((pos_j - pos_i) / self.radius)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def forward(self, x: Union[OptTensor, PairOptTensor], pos: Union[Tensor, PairTensor],
                edge_index: Adj, cond: Union[Tensor, None]) -> Tensor:

        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            if self.cat_pos:
                out = torch.cat([out, self.pe(pos[1])], dim=1)
            if self.use_film:
                cond = self.film(cond).reshape(-1, 2, out.shape[1])
                scale, bias = cond[:, 0], cond[:, 1]
                out = scale * out + bias
            out = self.global_nn(out)

        return out


class SAModule(torch.nn.Module):
    """Set Abstraction module for point cloud encoding."""

    def __init__(self, ratio, r, nn_dims, global_nn_dims=None, num_freqs=4, aggr='mean',
                 cat_pos=False, grouping_type='radius', use_film=False, cond_dim=512):
        super().__init__()
        self.grouping_type = grouping_type
        self.cat_pos = cat_pos
        self.ratio = ratio
        self.r = r
        self.use_film = use_film

        radius = r if grouping_type == 'radius' else 1.
        self.conv = PointNetConvPE(nn_dims, global_nn_dims, aggr=aggr, num_freqs=num_freqs,
                                   cat_pos=cat_pos, radius=radius, use_film=use_film, cond_dim=cond_dim)

    def forward(self, x, pos, batch, cond=None):
        idx = fps(pos, batch, ratio=self.ratio)
        if self.grouping_type == 'radius':
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=512)
        elif self.grouping_type == 'nearest':
            from torch_geometric.nn import nearest
            row = nearest(pos, pos[idx], batch, batch[idx])
            col = torch.arange(0, pos.shape[0], device=pos.device, dtype=torch.long)
        else:
            raise ValueError(f'Unknown grouping type: {self.grouping_type}')

        edge_index = torch.stack([col, row], dim=0)

        if self.use_film:
            cond = cond[batch[idx]]

        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index, cond)
        pos, batch = pos[idx], batch[idx]
        return [x, pos, batch]


class GlobalSAModule(torch.nn.Module):
    """Global Set Abstraction module."""

    def __init__(self, nn_dims, global_pool='max', num_freqs=4):
        super().__init__()
        nn_dims[0] += 3 * (2 * num_freqs)
        self.nn = MLP(nn_dims, plain_last=True, act=torch.nn.GELU(approximate='tanh'), norm=None)
        self.global_pool = global_pool
        self.pe = PositionalEncoder(3, num_freqs)

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, self.pe(pos)], dim=1))
        if self.global_pool == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return [x, pos, batch]


class Encoder(nn.Module):
    """
    Geometry encoder for point cloud embedding.

    Architecture:
    - 3-layer PointNet++ with set abstraction
    - Encodes point cloud to 512-dimensional feature vector
    - Used for demo retrieval via cosine similarity
    """

    def __init__(self, hidden_dim=512, num_freqs=4):
        super(Encoder, self).__init__()

        self.sa1_module = SAModule(0.1, 0.1, [3, 64, 64, 128], global_nn_dims=[128, 128, 128], num_freqs=num_freqs)
        self.sa2_module = SAModule(0.2, 0.2, [128 + 3, 128, 128, 256], global_nn_dims=[256, 256, 256], num_freqs=num_freqs)
        self.sa3_module = GlobalSAModule([256 + 3, 256, 512, hidden_dim], num_freqs=num_freqs)

    def encode_sample(self, data):
        """Encode a point cloud sample to a feature vector."""
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return x, pos, batch


class Decoder(nn.Module):
    """Decoder module (required for checkpoint loading compatibility)."""

    def __init__(self, nn_dims, num_freqs=4):
        super(Decoder, self).__init__()
        nn_dims[0] += 3 * (2 * num_freqs)
        self.pe = PositionalEncoder(3, num_freqs)
        self.nn = MLP(nn_dims)

    def decode_sample(self, queries, batch_queries, x):
        x = self.nn(torch.cat([x[batch_queries], self.pe(queries)], dim=1))
        return x


class AutoEncoder(pl.LightningModule):
    """
    AutoEncoder wrapper (required for checkpoint loading).

    Only the encoder is used for MT3 inference - the decoder is not used
    but must be present to load the trained checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config['pos_weight']))
        self.loss_buffer = torch.zeros(config['log_every_n_steps'])

        self.encoder = Encoder(hidden_dim=config['hidden_dim'], num_freqs=config['num_freqs'])
        self.decoder = Decoder([config['hidden_dim'] + 3, 512, 512, 512, 512, 1], num_freqs=config['num_freqs'])

        self.plotter = None


# ============================================================================
# Inference Functions
# ============================================================================

def load_encoder(path, model_name='geometry_encoder.ckpt', config_name='geometry_encoder_config.json'):
    """
    Load pre-trained geometry encoder for MT3 inference.

    Args:
        path: Path to directory containing checkpoint and config
        model_name: Name of checkpoint file (default: 'geometry_encoder.ckpt')
        config_name: Name of config file (default: 'geometry_encoder_config.json')

    Returns:
        encoder: Loaded encoder model ready for inference

    Example:
        >>> from thousand_tasks.core.globals import ASSETS_DIR
        >>> encoder = load_encoder(ASSETS_DIR)
        >>> encoder.eval()
    """
    with open(f"{path}/{config_name}", 'r') as f:
        config = json.load(f)
    auto_encoder = AutoEncoder.load_from_checkpoint(f'{path}/{model_name}', config=config, map_location='cpu')

    return auto_encoder.encoder


def get_pcd_embd(encoder, scene_state):
    """
    Encode a segmented point cloud into a 512-dimensional feature vector.

    This function:
    1. Extracts point cloud from SceneState
    2. Voxel downsamples to 5mm resolution
    3. Transforms to world frame
    4. Centers the point cloud
    5. Encodes to 512-dim vector

    Args:
        encoder: Pre-trained encoder model
        scene_state: SceneState with segmented RGB-D data

    Returns:
        embedding: 512-dimensional numpy array

    Example:
        >>> from thousand_tasks.core.utils.scene_state import SceneState
        >>> scene_state = SceneState.initialise_from_dict({...})
        >>> embedding = get_pcd_embd(encoder, scene_state)
        >>> print(embedding.shape)  # (512,)
    """
    pcd_o3d = scene_state.o3d_pcd
    # Consistent with the training data - 5mm voxel downsampling
    pcd_o3d = pcd_o3d.voxel_down_sample(0.005)
    pcd = np.asarray(pcd_o3d.points)

    T_WC = scene_state.T_WC
    # Transform to world frame
    pcd = np.matmul(T_WC[:3, :3], pcd.T).T + T_WC[:3, 3]
    # Center the point cloud
    pcd = pcd - np.mean(pcd, axis=0)

    encoder_device = next(encoder.parameters()).device
    with torch.no_grad():
        data = Data(x=None, pos=torch.tensor(pcd).float(), batch=torch.zeros(pcd.shape[0]).long())
        x, pos, batch = encoder.encode_sample(data.to(encoder_device))
    return x.squeeze().detach().cpu().numpy()
