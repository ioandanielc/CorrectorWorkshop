"""
model12: iterative message-passing corrector.

Why it exists — model9's ceiling, diagnosed on the SPH-loss task:
one violation-gated interaction round means displacements depend only on a
particle's own violating pairs. Non-violating asymmetry (which carries most of
the kernel-gradient signal) is invisible, and coordinated multi-neighbour
rearrangement is out of reach. Training saturates within ~1-2k iterations.

model12 changes three things:
1. L message-passing rounds with residual node states — information propagates
   L hops, receptive field ~ L * rd.
2. Smooth proximity attention: pairs within the attention radius rd get weight
   (1 - (d/rd)^2)^2, normalised per particle — smooth to zero at the cutoff,
   nonzero for every near pair, violating or not. The violation feature
   relu(rd - d) stays in the edge features so overlap depth is still explicit.
3. Optional periodic geometry: forward(x, rd, box=L) computes relative
   positions under the minimum image, so wrap-seam pairs are visible during
   periodic training (model9 is blind to them).

Calling convention: model(x, rd=attention_rd, box=None). rd is the ATTENTION
radius (the checkpoint's model config carries it as attention_rd) — not
necessarily the constraint rd of the loss.

Config keys: hidden_dim, num_layers, norm, activation, max_displacement.

Note: edges are materialised densely, (B, N, N, ·) — fine for tile-sized
clouds (N ~ 50-250); a sparsified neighbour list would be needed before
running whole point clouds through it.
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class CorrectorModel(nn.Module):
    uses_rd  = True
    uses_box = True

    def __init__(self, model_config, input_dim, initialization):
        super().__init__()
        H = model_config['hidden_dim']
        L = model_config.get('num_layers', 3)
        self.hidden_dim = H
        self.num_layers = L
        self.max_disp   = model_config.get('max_displacement', 0.06)

        act    = getattr(nn, model_config['activation'])
        use_ln = model_config.get('norm') == 'layer'

        def mlp(d_in):
            layers = [nn.Linear(d_in, H)]
            if use_ln:
                layers.append(nn.LayerNorm(H))
            layers += [act(), nn.Linear(H, H)]
            return nn.Sequential(*layers)

        edge_in = 2 * H + input_dim + 2       # h_i, h_j, rel_pos, dist, relu(rd-d)
        self.edge_mlps = nn.ModuleList([mlp(edge_in) for _ in range(L)])
        self.node_mlps = nn.ModuleList([mlp(H) for _ in range(L)])
        self.out_mlp   = nn.Sequential(nn.Linear(H, H), act(),
                                       nn.Linear(H, input_dim))
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
        if rd is None:
            raise ValueError("model12 requires rd (the attention radius)")
        B, N, D = x.shape

        rel = x.unsqueeze(2) - x.unsqueeze(1)              # (B, N, N, D)
        if box is not None:
            rel = rel - box * torch.round(rel / box)       # minimum image
        dist = rel.norm(dim=-1)                            # (B, N, N)
        eye  = torch.eye(N, dtype=torch.bool, device=x.device)

        # smooth proximity attention, zero at d >= rd, normalised per particle
        q = (dist / rd).clamp(max=1.0)
        w = ((1.0 - q ** 2) ** 2).masked_fill(eye, 0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)       # (B, N, N)

        geo = torch.cat([rel, dist.unsqueeze(-1),
                         torch.relu(rd - dist).unsqueeze(-1)], dim=-1)

        h = x.new_zeros(B, N, self.hidden_dim)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)
            e  = edge_mlp(torch.cat([hi, hj, geo], dim=-1))    # (B, N, N, H)
            msg = (w.unsqueeze(-1) * e).sum(dim=2)             # (B, N, H)
            h   = h + node_mlp(msg)                            # residual update

        return torch.tanh(self.out_mlp(h)) * self.max_disp

    def forward_sparse(self, x, edge_index, rd=None, box=None):
        """Weight-identical sparse version of forward() for a single cloud.

        Runs the exact same message passing as forward(), but only over the
        edges in edge_index. This is EXACT, not an approximation: the proximity
        weight (1 - (d/rd)^2)^2 is identically 0 at d >= rd, so pairs beyond the
        cutoff contribute nothing to either the messages or the per-particle
        normaliser. Dropping them removes the dense (N, N) materialisation and
        lets N reach the thousands (whole-cloud inference), which forward()
        cannot fit.

        Parameters
        ----------
        x          : (N, D) positions of one cloud
        edge_index : (2, E) long tensor; row 0 = receiver i, row 1 = sender j.
                     Must list every ordered pair (i, j), i != j, whose
                     minimum-image distance is < rd — both directions, no
                     self-loops. The caller builds this (e.g. a PBC cKDTree).
        rd, box    : same meaning as forward().
        """
        if rd is None:
            raise ValueError("model12 requires rd (the attention radius)")
        N = x.shape[0]
        dst, src = edge_index[0], edge_index[1]

        rel = x[dst] - x[src]                               # (E, D)
        if box is not None:
            rel = rel - box * torch.round(rel / box)        # minimum image
        dist = rel.norm(dim=-1)                             # (E,)

        # smooth proximity attention, normalised per receiver (matches forward())
        q   = (dist / rd).clamp(max=1.0)
        w   = (1.0 - q ** 2) ** 2                           # (E,) 0 at d>=rd
        deg = x.new_zeros(N).index_add_(0, dst, w)          # sum_j w over receivers
        w   = w / (deg[dst] + 1e-8)                         # (E,)

        geo = torch.cat([rel, dist.unsqueeze(-1),
                         torch.relu(rd - dist).unsqueeze(-1)], dim=-1)   # (E, D+2)

        h = x.new_zeros(N, self.hidden_dim)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e   = edge_mlp(torch.cat([h[dst], h[src], geo], dim=-1))     # (E, H)
            msg = x.new_zeros(N, self.hidden_dim).index_add_(
                0, dst, w.unsqueeze(-1) * e)                             # (N, H)
            h   = h + node_mlp(msg)

        return torch.tanh(self.out_mlp(h)) * self.max_disp
