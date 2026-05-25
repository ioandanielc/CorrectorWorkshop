"""
model6: Violation-aware edge network.

Replaces global self-attention on raw coordinates with physically motivated
edge features between all point pairs:
  - relative position:  xi - xj          (dim-D, translationally invariant)
  - distance:           ||xi - xj||       (scalar)
  - violation signal:   relu(rd - dist)   (scalar, explicit constraint feedback)

Per-point displacement is computed by:
  1. edge_mlp:    embed each (i,j) edge feature vector
  2. sum over j:  aggregate neighbour messages per point
  3. output_mlp:  map aggregated features → displacement vector

Properties:
- Translationally invariant by construction (uses relative positions)
- Rotationally invariant by construction (uses distances + relative positions
  which transform consistently under rotation)
- No DataProcessor dependency for invariance (though centering still applied upstream)
- ~34K parameters vs ~10.5M for model4 — appropriate for N=50, dim=2
- uses_rd = True — trainer passes rd as keyword argument

Config keys: hidden_dim, norm ('layer' or 'none'), activation
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd = True  # signals trainer to pass rd to forward()

    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = model_config['hidden_dim']
        self.norm_type = model_config['norm']   # 'layer' or 'none'

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

        # Embeds each directed edge (i→j) feature vector
        self.edge_mlp = make_mlp(edge_dim, self.hidden_dim, self.hidden_dim)

        # Maps aggregated per-point context → displacement
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
        # x:  (B, N, dim) — centroid removed by DataProcessor
        # rd: scalar tensor — Poisson disk radius
        if rd is None:
            raise ValueError("model6 requires rd as input")

        B, N, D = x.shape

        # --- Build edge features (B, N, N, dim+2) ---
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, D)  point i repeated
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, D)  point j repeated
        rel_pos = xi - xj                             # (B, N, N, D)  relative position i→j

        dist = rel_pos.norm(dim=-1, keepdim=True)    # (B, N, N, 1)
        violation = torch.relu(rd - dist)            # (B, N, N, 1)  > 0 only if too close

        edge_feat = torch.cat([rel_pos, dist, violation], dim=-1)  # (B, N, N, D+2)

        # --- Embed edges (B, N, N, hidden_dim) ---
        edge_emb = self.edge_mlp(edge_feat)

        # --- Zero out self-edges (i==j) before aggregation ---
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        edge_emb = edge_emb * (~eye).unsqueeze(0).unsqueeze(-1)   # broadcast over B, hidden

        # --- Aggregate: sum over j → (B, N, hidden_dim) ---
        agg = edge_emb.sum(dim=2)

        # --- Per-point displacement (B, N, dim) ---
        displacement = self.output_mlp(agg)

        # return_attention_maps kept for API compatibility — returns None
        if return_attention_maps:
            return displacement, None
        return displacement
