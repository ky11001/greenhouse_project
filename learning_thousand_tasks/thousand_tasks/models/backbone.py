import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from thousand_tasks.retrieval.auto_encoder import SAModule, PositionalEncoder


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_film = config.use_film

        self.sa1_module = SAModule(ratio=config.ratio_1,
                                   r=config.radius_1,
                                   nn_dims=config.locla_nn_dims_1,
                                   global_nn_dims=config.global_nn_dims_1,
                                   num_freqs=config.num_freqs_pt,
                                   use_film=config.use_film,
                                   cond_dim=config.lang_dim,
                                   )

        self.sa2_module = SAModule(ratio=config.ratio_2,
                                   r=config.radius_2,
                                   nn_dims=config.locla_nn_dims_2,
                                   global_nn_dims=config.global_nn_dims_2,
                                   num_freqs=config.num_freqs_pt,
                                   cat_pos=config.cat_pos,
                                   use_film=config.use_film,
                                   cond_dim=config.lang_dim,
                                   )

        self.pe_sin = PositionalEncoder(d_input=3,
                                        n_freqs=config.num_freqs_pe,
                                        add_original_x=True)

        self.positional_encoder = nn.Sequential(
            self.pe_sin,
            nn.Linear(in_features=self.pe_sin.d_output, out_features=config.hidden_dim))

    def forward(self, data, cond=None):
        x, pos, batch = self.sa1_module(data.rgb, data.pos, data.batch, cond)
        x, pos, batch = self.sa2_module(x, pos, batch, cond)

        pos = self.positional_encoder(pos)

        x_batched, _ = to_dense_batch(x, batch)
        pos_batched, _ = to_dense_batch(pos, batch)

        return x_batched, pos_batched


def build_backbone(args):
    return Backbone(args)
