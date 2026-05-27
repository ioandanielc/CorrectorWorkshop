"""
model9: violation-weighted edge network with two targeted improvements over model7/8:

1. Deeper edge MLP (3 hidden layers instead of 2)
   Gives the model more capacity to learn complex local conflict geometry.
   Multi-point clusters (3+ mutually-violating points) require higher-order
   reasoning about local structure that 2-layer MLPs under-represent.

2. Output displacement clamping via tanh * max_displacement
   Replaces the raw linear output layer. Hard-limits per-step displacement,
   which prevents:
     - Runaway corrections in early training
     - Overcorrections in packed clouds that cascade into new violations
     - Max disp / rd values of 1.7-3.4x seen in model7/8 eval
   Forces the model to be efficient (correct precisely, not excessively).

Architecture otherwise identical to model7:
- Translationally invariant (relative positions)
- Violation-weighted aggregation (non-violating neighbours contribute zero)
- uses_rd = True

Config keys: hidden_dim, norm, activation, max_displacement
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd = True

    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.input_dim    = input_dim
        self.hidden_dim   = model_config['hidden_dim']
        self.norm_type    = model_config['norm']
        self.max_disp     = model_config.get('max_displacement', 0.06)  # ~1.2 * rd default

        activation_cls = getattr(nn, model_config['activation'])
        edge_dim = input_dim + 2  # rel_pos (dim) + distance (1) + violation (1)

        def make_mlp(*dims):
            layers = []
            for in_d, out_d in zip(dims, dims[1:]):
                layers.append(nn.Linear(in_d, out_d))
                if self.norm_type == 'layer':
                    layers.append(nn.LayerNorm(out_d))
                layers.append(activation_cls())
            return nn.Sequential(*layers)

        # Configurable-depth edge MLP (default 3, was 2 in model7/8)
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
            raise ValueError("model9 requires rd as input")

        B, N, D = x.shape

        # --- Edge features (B, N, N, dim+2) ---
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)
        rel_pos   = xi - xj
        dist      = rel_pos.norm(dim=-1, keepdim=True)
        violation = torch.relu(rd - dist)

        edge_feat = torch.cat([rel_pos, dist, violation], dim=-1)

        # --- Embed edges ---
        edge_emb = self.edge_mlp(edge_feat)

        # --- Zero self-edges ---
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        edge_emb  = edge_emb  * (~eye).unsqueeze(0).unsqueeze(-1)
        violation = violation * (~eye).unsqueeze(0).unsqueeze(-1)

        # --- Violation-weighted aggregation ---
        viol_sum = violation.sum(dim=2, keepdim=True)
        weights  = violation / (viol_sum + 1e-8)
        agg = (edge_emb * weights).sum(dim=2)

        # --- Clamped displacement output ---
        raw_disp    = self.output_mlp(agg)
        displacement = torch.tanh(raw_disp) * self.max_disp

        if return_attention_maps:
            return displacement, weights.squeeze(-1)
        return displacement
