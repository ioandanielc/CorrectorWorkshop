"""
model10: model9 + self-feature (point position concatenated to aggregated message).

The only change from model9: after violation-weighted aggregation produces
agg_i (shape hidden_dim), we concatenate x_i (the point's own normalized
position, shape input_dim) before the output MLP.

Motivation: the aggregated message is purely relational — it encodes which
neighbours violate and by how much, but carries no information about where
point i sits in the cloud. Near-boundary points have asymmetric neighbourhoods
that look identical to interior points after aggregation; x_i breaks that
symmetry and lets the model learn boundary-aware corrections.

Note: x is already centered and normalized by DataProcessor.make_invariant,
so x_i = 0 at the centroid; values elsewhere are meaningful relative to the
cloud extent.

Config keys: hidden_dim, edge_depth, norm, activation, max_displacement
             (identical to model9 — drop-in replacement for comparison)
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd = True

    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = model_config['hidden_dim']
        self.norm_type  = model_config['norm']
        self.max_disp   = model_config.get('max_displacement', 0.06)

        activation_cls = getattr(nn, model_config['activation'])
        edge_dim = input_dim + 2  # rel_pos + dist + violation

        def make_mlp(*dims):
            layers = []
            for in_d, out_d in zip(dims, dims[1:]):
                layers.append(nn.Linear(in_d, out_d))
                if self.norm_type == 'layer':
                    layers.append(nn.LayerNorm(out_d))
                layers.append(activation_cls())
            return nn.Sequential(*layers)

        edge_depth = model_config.get('edge_depth', 3)
        self.edge_mlp = make_mlp(edge_dim, *([self.hidden_dim] * edge_depth))

        # Output MLP receives [agg (hidden_dim) || x_i (input_dim)]
        self.output_mlp = nn.Sequential(
            *make_mlp(self.hidden_dim + input_dim, self.hidden_dim),
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
            raise ValueError("model10 requires rd as input")

        B, N, D = x.shape

        # --- Edge features ---
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)
        rel_pos   = xi - xj
        dist      = rel_pos.norm(dim=-1, keepdim=True)
        violation = torch.relu(rd - dist)

        edge_feat = torch.cat([rel_pos, dist, violation], dim=-1)
        edge_emb  = self.edge_mlp(edge_feat)

        # --- Zero self-edges ---
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        edge_emb  = edge_emb  * (~eye).unsqueeze(0).unsqueeze(-1)
        violation = violation * (~eye).unsqueeze(0).unsqueeze(-1)

        # --- Violation-weighted aggregation ---
        viol_sum = violation.sum(dim=2, keepdim=True)
        weights  = violation / (viol_sum + 1e-8)
        agg = (edge_emb * weights).sum(dim=2)  # (B, N, hidden_dim)

        # --- Concatenate self-feature and decode ---
        agg_with_self = torch.cat([agg, x], dim=-1)  # (B, N, hidden_dim + input_dim)
        raw_disp      = self.output_mlp(agg_with_self)
        displacement  = torch.tanh(raw_disp) * self.max_disp

        if return_attention_maps:
            return displacement, weights.squeeze(-1)
        return displacement
