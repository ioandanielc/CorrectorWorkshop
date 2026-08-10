"""
GNS-style corrector — the learned-simulator baseline.

Encoder-processor-decoder over a radius graph, after Sanchez-Gonzalez et al.
(2020), "Learning to Simulate Complex Physics with Graph Networks". Adapted to
the static correction task: GNS consumes a velocity history and predicts
accelerations for a rollout, whereas here there is no history and one pass
predicts a displacement, so the node latents start at zero (as in model12) and
only the geometry carries signal. Everything that defines GNS as an architecture
is kept: a persistent edge latent updated every round, residual updates on both
edge and node states, and plain summation over neighbours.

This is the closest published architecture to model12, which makes it the
sharpest baseline in the suite. The two differ in exactly two mechanisms:

    message weight   model12: fixed (1-q^2)^2 SPH kernel, normalised per particle
                     GNS:     learned, carried in a persistent edge latent
    aggregation      model12: kernel-weighted mean  |  GNS: unweighted sum
    neighbourhood    both:    radius graph at cutoff_rd, minimum-image

so model12 can be read as "GNS with the SPH kernel written into the message
weights". One consequence worth noting: model12's weight decays smoothly to zero
at the cutoff, while a hard radius mask makes the GNS message sum discontinuous
in the particle positions as pairs enter and leave the neighbourhood.

Config keys: hidden_dim, num_layers, norm, activation, max_displacement, cutoff_rd.
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

        self.edge_encoder = mlp(input_dim + 1)                             # rel, dist -> edge latent
        self.edge_mlps = nn.ModuleList([mlp(3 * H) for _ in range(L)])     # [e, h_i, h_j] -> edge update
        self.node_mlps = nn.ModuleList([mlp(2 * H) for _ in range(L)])     # [h_i, sum_j e] -> node update
        self.decoder = nn.Sequential(nn.Linear(H, H), act(), nn.Linear(H, input_dim))
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
            raise ValueError("gns requires rd (the connectivity radius)")
        B, N, D = x.shape

        rel = x.unsqueeze(2) - x.unsqueeze(1)              # (B, N, N, D)
        if box is not None:
            rel = rel - box * torch.round(rel / box)       # minimum image
        dist = rel.norm(dim=-1)                            # (B, N, N)
        eye  = torch.eye(N, dtype=torch.bool, device=x.device)

        # radius graph: a hard 0/1 mask, unlike model12's smooth kernel weight
        adj = ((dist < rd) & ~eye).to(x.dtype).unsqueeze(-1)   # (B, N, N, 1)

        e = self.edge_encoder(torch.cat([rel, dist.unsqueeze(-1)], dim=-1))  # (B, N, N, H)
        h = x.new_zeros(B, N, self.hidden_dim)

        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)
            e  = e + edge_mlp(torch.cat([e, hi, hj], dim=-1))   # residual edge state
            agg = (adj * e).sum(dim=2)                          # unweighted sum over neighbours
            h  = h + node_mlp(torch.cat([h, agg], dim=-1))      # residual node state

        return torch.tanh(self.decoder(h)) * self.max_disp

    def forward_sparse(self, x, edge_index, rd=None, box=None):
        """Weight-identical sparse version of forward() for a single cloud.

        Exact, not an approximation: the radius graph is a hard 0/1 mask, so pairs
        beyond the cutoff contribute nothing to any aggregation and their edge states
        are never read. Dropping them removes the dense (N, N) materialisation, which
        is what makes whole-cloud inference possible at all — at N=2500 the dense path
        needs ~8 GB per round.

        Parameters
        ----------
        x          : (N, D) positions of one cloud
        edge_index : (2, E) long; row 0 = receiver i, row 1 = sender j. Every ordered
                     pair within the connectivity radius, both directions, no
                     self-loops — the same contract as model12.forward_sparse.
        rd, box    : same meaning as forward().
        """
        if rd is None:
            raise ValueError("gns requires rd (the connectivity radius)")
        N = x.shape[0]
        dst, src = edge_index[0], edge_index[1]

        rel = x[dst] - x[src]                                   # (E, D)
        if box is not None:
            rel = rel - box * torch.round(rel / box)
        dist = rel.norm(dim=-1)                                 # (E,)

        e = self.edge_encoder(torch.cat([rel, dist.unsqueeze(-1)], dim=-1))   # (E, H)
        h = x.new_zeros(N, self.hidden_dim)

        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e = e + edge_mlp(torch.cat([e, h[dst], h[src]], dim=-1))
            agg = x.new_zeros(N, self.hidden_dim).index_add_(0, dst, e)
            h = h + node_mlp(torch.cat([h, agg], dim=-1))

        return torch.tanh(self.decoder(h)) * self.max_disp
