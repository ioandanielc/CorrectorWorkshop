"""
DGCNN corrector — the strong architecture baseline.

EdgeConv over a *dynamic* k-nearest-neighbour graph: each round rebuilds the
graph in the current feature space, forms [h_i, h_j - h_i] per edge, and
aggregates with max. It has local structure and message passing, so it differs
from model12 in exactly the components the architecture bridge isolates:

    graph        kNN in feature space   (model12: radius graph at cutoff_rd)
    edge weight  learned                (model12: fixed (1-q^2)^2 SPH kernel)
    aggregation  max                    (model12: kernel-weighted sum)

Periodicity is NOT one of those differences: the first round's neighbour search
and its relative positions both use the minimum image, so this baseline gets the
same geometric information model12 does. `rd` is accepted and ignored — a kNN
graph has no cutoff, and taking one would stop it being DGCNN — but the argument
must exist because the trainer only passes `box` to models declaring `uses_rd`.

Matched to model12 on everything the comparison is not about: same constructor
signature, same bounded `tanh * max_displacement` head, same parameter budget.

Config keys: hidden_dim, num_layers, k, norm, activation, max_displacement.
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd  = True    # accepted and ignored; required for the trainer to pass box
    uses_box = True

    def __init__(self, model_config, input_dim, initialization):
        super().__init__()
        H = model_config['hidden_dim']
        L = model_config.get('num_layers', 3)
        self.hidden_dim = H
        self.num_layers = L
        self.k          = model_config.get('k', 8)
        self.max_disp   = model_config.get('max_displacement', 0.06)

        act    = getattr(nn, model_config['activation'])
        use_ln = model_config.get('norm') == 'layer'

        def mlp(d_in):
            layers = [nn.Linear(d_in, H)]
            if use_ln:
                layers.append(nn.LayerNorm(H))
            layers += [act(), nn.Linear(H, H)]
            return nn.Sequential(*layers)

        # round 0 consumes coordinates ([x_i, rel_ij]); later rounds consume features
        self.edge_mlps = nn.ModuleList(
            [mlp(2 * input_dim)] + [mlp(2 * H) for _ in range(L - 1)])
        self.out_mlp = nn.Sequential(nn.Linear(H, H), act(), nn.Linear(H, input_dim))
        self._initialize(initialization)

    def _initialize(self, initialization):
        init_fn = {
            'xavier_uniform':  init.xavier_uniform_,
            'xavier_normal':   init.xavier_normal_,
            'kaiming_uniform': init.kaiming_uniform_,
            'kaiming_normal':  init.kaiming_normal_,
        }[initialization]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_fn(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x, rd=None, box=None):
        B, N, D = x.shape
        k = min(self.k, N - 1)
        eye = torch.eye(N, dtype=torch.bool, device=x.device)

        h = x
        for i, edge_mlp in enumerate(self.edge_mlps):
            # rel_ij = h_j - h_i; on round 0 these are positions, so wrap them
            rel = h.unsqueeze(1) - h.unsqueeze(2)          # (B, N, N, C) sender - receiver
            if i == 0 and box is not None:
                rel = rel - box * torch.round(rel / box)   # minimum image

            # dynamic graph: k nearest in the CURRENT feature space
            dist = rel.norm(dim=-1).masked_fill(eye, float('inf'))   # (B, N, N)
            idx  = dist.topk(k, dim=-1, largest=False).indices       # (B, N, k)

            gather = idx.unsqueeze(-1).expand(-1, -1, -1, rel.shape[-1])
            rel_k  = rel.gather(2, gather)                            # (B, N, k, C)
            hi_k   = h.unsqueeze(2).expand(-1, -1, k, -1)             # (B, N, k, C)

            e = edge_mlp(torch.cat([hi_k, rel_k], dim=-1))            # (B, N, k, H)
            h = e.max(dim=2).values                                   # max aggregation

        return torch.tanh(self.out_mlp(h)) * self.max_disp
