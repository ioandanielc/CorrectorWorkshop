"""
model11: variable-rd violation-weighted edge network.

Extends model9 (models/fixed_rd/model9.py) to generalize across rd instead of being
trained/deployed at one fixed rd. Two changes to the architecture:

1. rd-normalized edge features
   rel_pos, dist and violation are all divided by rd before the edge MLP (equivalently:
   solve the problem at rd=1 in rescaled coordinates). A Poisson-disk problem at radius
   rd is geometrically identical to the same problem at radius 1 after dividing every
   length by rd, so this gives the network a strong inductive bias for the ~100x rd
   range (0.01-1.0) it's trained on, rather than asking it to learn the scale
   relationship purely from data.

2. rd embedding
   log10(rd) is passed through a small MLP and concatenated onto every edge feature
   vector. This gives the network a channel for whatever *isn't* scale-invariant about
   the problem (e.g. the fixed [0,1] domain boundary mattering proportionally more as
   rd grows) on top of the normalized geometric features.

Output displacement is predicted in rd-normalized units (tanh-clamped to
max_displacement_factor, replacing model9's absolute max_displacement) and then
multiplied back by rd.

rd is per-batch (one scalar shared by the whole batch, not per-cloud), same calling
convention as model9: uses_rd = True, forward(x, rd=<0-dim tensor>).

Config keys: hidden_dim, norm, activation, edge_depth, max_displacement_factor, rd_embed_dim
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd = True

    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.input_dim        = input_dim
        self.hidden_dim       = model_config['hidden_dim']
        self.norm_type        = model_config['norm']
        self.max_disp_factor  = model_config.get('max_displacement_factor', 1.2)
        self.rd_embed_dim     = model_config.get('rd_embed_dim', 16)

        activation_cls = getattr(nn, model_config['activation'])
        edge_dim = input_dim + 2 + self.rd_embed_dim  # rel_pos/rd (dim) + dist/rd (1) + viol/rd (1) + rd_embed

        def make_mlp(*dims):
            layers = []
            for in_d, out_d in zip(dims, dims[1:]):
                layers.append(nn.Linear(in_d, out_d))
                if self.norm_type == 'layer':
                    layers.append(nn.LayerNorm(out_d))
                layers.append(activation_cls())
            return nn.Sequential(*layers)

        # rd embedding: log10(rd) (scalar) -> rd_embed_dim
        self.rd_embed = make_mlp(1, self.rd_embed_dim, self.rd_embed_dim)

        # Configurable-depth edge MLP (default 3, same as model9)
        edge_depth = model_config.get('edge_depth', 3)
        self.edge_mlp = make_mlp(edge_dim, *([self.hidden_dim] * edge_depth))

        # Output MLP: same depth, but final layer replaced by tanh-clamped output
        self.output_mlp = nn.Sequential(
            *make_mlp(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, input_dim),
        )

        self._initialize_weights(initialization)

    def _initialize_weights(self, initialization: str):
        init_fn = {
            'xavier_uniform':  init.xavier_uniform_,
            'xavier_normal':   init.xavier_normal_,
            'kaiming_uniform': init.kaiming_uniform_,
            'kaiming_normal':  init.kaiming_normal_,
        }.get(initialization)
        if init_fn is None:
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_fn(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x, rd=None, return_attention_maps=False):
        if rd is None:
            raise ValueError("model11 requires rd as input")

        B, N, D = x.shape
        H = self.hidden_dim

        # --- Pairwise distances (raw units, for the violation mask) ---
        rel_pos_raw = x.unsqueeze(2) - x.unsqueeze(1)          # (B, N, N, D)
        dist_raw    = rel_pos_raw.norm(dim=-1)                   # (B, N, N)
        eye         = torch.eye(N, dtype=torch.bool, device=x.device)

        # --- Sparse: only process violating, non-self pairs ---
        viol_mask = (dist_raw < rd) & ~eye                       # (B, N, N)
        b_idx, i_idx, j_idx = viol_mask.nonzero(as_tuple=True)   # (M,)

        agg = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

        # rd embedding: one vector for the whole batch (rd is per-batch, not per-cloud)
        rd_log   = torch.log10(rd).reshape(1, 1)                 # (1, 1)
        rd_emb   = self.rd_embed(rd_log).reshape(-1)              # (rd_embed_dim,)

        if b_idx.numel() > 0:
            # rd-normalized edge features for violating pairs only
            rel_v = rel_pos_raw[b_idx, i_idx, j_idx] / rd        # (M, D)
            d_v   = (dist_raw   [b_idx, i_idx, j_idx, None]) / rd  # (M, 1)
            v_v   = torch.relu(1.0 - d_v)                         # (M, 1) == relu(rd - dist)/rd
            rd_v  = rd_emb.unsqueeze(0).expand(b_idx.numel(), -1)  # (M, rd_embed_dim)
            edge_emb = self.edge_mlp(
                torch.cat([rel_v, d_v, v_v, rd_v], dim=-1))      # (M, H)

            # Violation-weighted aggregation via scatter_add
            flat_i   = (b_idx * N + i_idx).unsqueeze(-1)         # (M, 1)

            viol_sum = torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
            viol_sum.scatter_add_(0, flat_i, v_v)

            w = v_v / (viol_sum[b_idx * N + i_idx] + 1e-8)       # (M, 1)

            agg_flat = torch.zeros(B * N, H, device=x.device, dtype=x.dtype)
            agg_flat.scatter_add_(0, flat_i.expand(-1, H), edge_emb * w)
            agg = agg_flat.view(B, N, H)

        # --- Clamped displacement output, in rd-normalized units, scaled back by rd ---
        displacement = torch.tanh(self.output_mlp(agg)) * self.max_disp_factor * rd

        if return_attention_maps:
            viol_sum_full = (viol_sum.view(B, N, 1) if b_idx.numel() > 0
                             else torch.zeros(B, N, 1, device=x.device, dtype=x.dtype))
            viol_map_full = torch.relu(1.0 - dist_raw / rd)
            weights = viol_map_full / (viol_sum_full + 1e-8) * (~eye).to(x.dtype)
            return displacement, weights
        return displacement
