"""
model7: Violation-weighted edge aggregation.

Changes vs model6:
- Aggregation step replaces uniform sum with violation-weighted messages.
  Each neighbour j's edge embedding is scaled by its violation depth before
  summing into point i's context vector:

      weights[i,j] = violation[i,j] / (sum_j violation[i,j] + eps)
      agg[i]       = sum_j (weights[i,j] * edge_emb[i,j])

  Non-violating neighbours (depth = 0) contribute exactly zero.
  Severely violating neighbours dominate proportionally to their depth.

Why this matters:
  In model6, a point surrounded by 49 perfectly-legal neighbours and 1
  violating one averages the single important signal across all 49 others,
  diluting the gradient. Here, only the violating neighbour contributes —
  the aggregated context is purely about what needs fixing.

  Expected effect: cleaner gradient signal in late training; less spurious
  displacement of already-legal points; correction_eff closer to the 2/(N-1)
  theoretical ceiling.

Properties (unchanged from model6):
- Translationally invariant by construction (relative positions)
- Rotationally invariant (distances + relative positions)
- Permutation invariant (weighted sum over j)
- uses_rd = True — trainer passes rd as keyword argument

Config keys: hidden_dim, norm ('layer' or 'none'), activation
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd = True

    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = model_config['hidden_dim']
        self.norm_type = model_config['norm']

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

        self.edge_mlp = make_mlp(edge_dim, self.hidden_dim, self.hidden_dim)

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
        # x:  (B, N, dim)
        # rd: scalar tensor
        if rd is None:
            raise ValueError("model7 requires rd as input")

        B, N, D = x.shape

        # --- Build edge features (B, N, N, dim+2) ---
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, D)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, D)
        rel_pos   = xi - xj                           # (B, N, N, D)
        dist      = rel_pos.norm(dim=-1, keepdim=True)  # (B, N, N, 1)
        violation = torch.relu(rd - dist)             # (B, N, N, 1)  > 0 only if too close

        edge_feat = torch.cat([rel_pos, dist, violation], dim=-1)  # (B, N, N, D+2)

        # --- Embed edges ---
        edge_emb = self.edge_mlp(edge_feat)           # (B, N, N, hidden_dim)

        # --- Zero out self-edges (i == j) ---
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        edge_emb  = edge_emb  * (~eye).unsqueeze(0).unsqueeze(-1)
        violation = violation * (~eye).unsqueeze(0).unsqueeze(-1)

        # --- Violation-weighted aggregation ---
        # weights[b,i,j] = violation[b,i,j] / (sum_j violation[b,i,j] + eps)
        # Points with zero total violation get weight=0 everywhere → agg = 0 → no displacement.
        viol_sum = violation.sum(dim=2, keepdim=True)          # (B, N, 1, 1)
        weights  = violation / (viol_sum + 1e-8)               # (B, N, N, 1)  sums to ~1 per i
        agg = (edge_emb * weights).sum(dim=2)                  # (B, N, hidden_dim)

        # --- Per-point displacement ---
        displacement = self.output_mlp(agg)           # (B, N, dim)

        if return_attention_maps:
            return displacement, weights.squeeze(-1)   # expose violation weights as "attention"
        return displacement
