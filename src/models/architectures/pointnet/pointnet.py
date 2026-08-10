"""
PointNet corrector — architecture ablation baseline.

Per-point encoder, a single global max-pooled feature, and a per-point head that
sees only [its own feature, the global feature]. Permutation-equivariant like
model12, but with **no pairwise term at all**: a point's displacement depends on
its own position and a summary of the whole cloud, never on which neighbours are
close. It is the no-local-geometry floor of the architecture bridge — the control
that shows the task needs relative geometry, not just per-point capacity.

Because there is no pairwise term there is nowhere for the minimum image to
apply, so `uses_box = False` and the network sees raw coordinates on the torus.
That is a real property of the architecture, not a handicap imposed here.

Matched to model12 on everything the comparison is not about: same constructor
signature, same bounded `tanh * max_displacement` head, same parameter budget.

Config keys: hidden_dim, norm, activation, max_displacement.
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd  = False
    uses_box = False

    def __init__(self, model_config, input_dim, initialization):
        super().__init__()
        H = model_config['hidden_dim']
        self.hidden_dim = H
        self.max_disp   = model_config.get('max_displacement', 0.06)

        act    = getattr(nn, model_config['activation'])
        use_ln = model_config.get('norm') == 'layer'

        def block(d_in, d_out, n_hidden):
            layers = [nn.Linear(d_in, H)]
            if use_ln:
                layers.append(nn.LayerNorm(H))
            for _ in range(n_hidden):
                layers += [act(), nn.Linear(H, H)]
            layers += [act(), nn.Linear(H, d_out)]
            return nn.Sequential(*layers)

        self.encoder = block(input_dim, H, 1)      # per-point features
        self.head    = block(2 * H, input_dim, 1)  # [own feature, global feature] -> displacement
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
        h = self.encoder(x)                                  # (B, N, H)
        g = h.max(dim=1, keepdim=True).values                # (B, 1, H) permutation-invariant
        h = torch.cat([h, g.expand(-1, x.shape[1], -1)], -1)  # (B, N, 2H)
        return torch.tanh(self.head(h)) * self.max_disp
