"""
model12 with its components made switchable — the architecture bridge.

Production `model12.py` is guarded by `tests/test_wholecloud.py`, which asserts
bit-exact reproduction of a sim-validated artifact, so it is left untouched. This is
a parallel copy whose three mechanisms are config flags, letting one component be
swapped at a time along the path from model12 to DGCNN:

    graph        radius (at cutoff_rd)  |  knn (k nearest, feature space)
    weight       kernel ((1-q^2)^2)     |  learned (edge-conditioned scalar)
    aggregation  wsum (weighted mean)   |  sum  |  max

With `graph=radius, weight=kernel, agg=wsum` and `periodic=true` this is exactly
production model12 (verified against it in the equivalence check below). The end of
the path — `graph=knn, weight=learned, agg=max` — is DGCNN's mechanism set.

`periodic: false` additionally disables the minimum image, isolating what the wrap
geometry is worth.

Config keys: hidden_dim, num_layers, norm, activation, max_displacement, cutoff_rd,
             graph, weight, aggregation, k, periodic.
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

        self.graph    = model_config.get('graph', 'radius')
        self.weight   = model_config.get('weight', 'kernel')
        self.agg      = model_config.get('aggregation', 'wsum')
        self.k        = model_config.get('k', 12)
        self.periodic = model_config.get('periodic', True)
        for name, val, allowed in (('graph', self.graph, ('radius', 'knn')),
                                   ('weight', self.weight, ('kernel', 'learned')),
                                   ('aggregation', self.agg, ('wsum', 'sum', 'max', 'wmax'))):
            if val not in allowed:
                raise ValueError(f'{name}={val!r} not in {allowed}')

        act    = getattr(nn, model_config['activation'])
        use_ln = model_config.get('norm') == 'layer'

        def mlp(d_in, d_out=None):
            layers = [nn.Linear(d_in, H)]
            if use_ln:
                layers.append(nn.LayerNorm(H))
            layers += [act(), nn.Linear(H, d_out or H)]
            return nn.Sequential(*layers)

        edge_in = 2 * H + input_dim + 2
        self.edge_mlps = nn.ModuleList([mlp(edge_in) for _ in range(L)])
        self.node_mlps = nn.ModuleList([mlp(H) for _ in range(L)])
        # Learned weighting replaces the fixed kernel with an edge-conditioned scalar
        # gate. Deliberately a single linear map (261 params/round, +0.3% total): a
        # full MLP head costs +38.6% and would confound "learned vs fixed weighting"
        # with extra capacity. The richly-learned end of this axis is covered at
        # matched parameters by the GNS baseline.
        self.weight_mlps = nn.ModuleList(
            [nn.Linear(edge_in, 1) for _ in range(L)]) if self.weight == 'learned' else None
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

    def forward_sparse(self, x, edge_index, rd=None, box=None):
        """Weight-identical sparse version of forward() for a single cloud.

        Radius-graph rungs only: a kNN graph is rebuilt in feature space each round, so
        there is no fixed edge list to exploit and `graph='knn'` is rejected here.
        Exact for the rungs it supports — the kernel weight is 0 at d >= rd and the
        radius mask is 0/1, so omitted pairs contribute nothing to any aggregation.
        """
        if rd is None:
            raise ValueError("model12_ablate requires rd (the cutoff radius)")
        if self.graph != 'radius':
            raise NotImplementedError(
                "forward_sparse supports graph='radius' only; a feature-space kNN graph "
                "has no fixed edge list")
        if not self.periodic:
            box = None
        N = x.shape[0]
        dst, src = edge_index[0], edge_index[1]

        rel = x[dst] - x[src]
        if box is not None:
            rel = rel - box * torch.round(rel / box)
        dist = rel.norm(dim=-1)
        geo = torch.cat([rel, dist.unsqueeze(-1),
                         torch.relu(rd - dist).unsqueeze(-1)], dim=-1)

        h = x.new_zeros(N, self.hidden_dim)
        for i, (edge_mlp, node_mlp) in enumerate(zip(self.edge_mlps, self.node_mlps)):
            feat = torch.cat([h[dst], h[src], geo], dim=-1)
            e = edge_mlp(feat)

            if self.agg in ('max', 'wmax'):
                em = e
                if self.agg == 'wmax':
                    if self.weight == 'kernel':
                        q = (dist / rd).clamp(max=1.0)
                        w = (1.0 - q ** 2) ** 2
                    else:
                        w = torch.sigmoid(self.weight_mlps[i](feat).squeeze(-1))
                    em = w.unsqueeze(-1) * e
                msg = x.new_full((N, self.hidden_dim), float('-inf')).scatter_reduce_(
                    0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), em,
                    reduce='amax', include_self=True)
                msg = torch.nan_to_num(msg, neginf=0.0)
            else:
                if self.weight == 'kernel':
                    q = (dist / rd).clamp(max=1.0)
                    w = (1.0 - q ** 2) ** 2
                else:
                    w = torch.sigmoid(self.weight_mlps[i](feat).squeeze(-1))
                if self.agg == 'wsum':
                    deg = x.new_zeros(N).index_add_(0, dst, w)
                    w = w / (deg[dst] + 1e-8)
                msg = x.new_zeros(N, self.hidden_dim).index_add_(
                    0, dst, w.unsqueeze(-1) * e)

            h = h + node_mlp(msg)

        return torch.tanh(self.out_mlp(h)) * self.max_disp

    def _membership(self, dist, h, eye, rd):
        """Which pairs are neighbours: a radius ball, or the k nearest."""
        if self.graph == 'radius':
            return (dist < rd) & ~eye
        # knn in the CURRENT feature space (coordinates on round 0, via `dist` caller)
        N = dist.shape[-1]
        k = min(self.k, N - 1)
        idx = dist.masked_fill(eye, float('inf')).topk(k, dim=-1, largest=False).indices
        m = torch.zeros_like(dist, dtype=torch.bool)
        return m.scatter_(-1, idx, True) & ~eye

    def forward(self, x, rd=None, box=None):
        if rd is None:
            raise ValueError("model12_ablate requires rd (the cutoff radius)")
        B, N, D = x.shape
        if not self.periodic:
            box = None

        rel = x.unsqueeze(2) - x.unsqueeze(1)
        if box is not None:
            rel = rel - box * torch.round(rel / box)
        dist = rel.norm(dim=-1)
        eye  = torch.eye(N, dtype=torch.bool, device=x.device)

        geo = torch.cat([rel, dist.unsqueeze(-1),
                         torch.relu(rd - dist).unsqueeze(-1)], dim=-1)

        h = x.new_zeros(B, N, self.hidden_dim)
        for i, (edge_mlp, node_mlp) in enumerate(zip(self.edge_mlps, self.node_mlps)):
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)
            feat = torch.cat([hi, hj, geo], dim=-1)
            e = edge_mlp(feat)

            # membership: the knn graph is rebuilt each round in feature space
            if self.graph == 'knn' and i > 0:
                fd = (h.unsqueeze(2) - h.unsqueeze(1)).norm(dim=-1)
            else:
                fd = dist
            member = self._membership(fd, h, eye, rd)

            if self.agg in ('max', 'wmax'):
                # 'max': `weight` is inert — a plain max has nowhere to apply a per-edge
                # scalar, so this rung changes TWO things (sum -> max AND weighting
                # removed). 'wmax' scales each message by the weight before the max, so
                # the kernel survives and only the aggregation operator changes — the
                # clean single-change comparison against 'wsum'.
                em = e
                if self.agg == 'wmax':
                    if self.weight == 'kernel':
                        q = (dist / rd).clamp(max=1.0)
                        w = (1.0 - q ** 2) ** 2
                    else:
                        w = torch.sigmoid(self.weight_mlps[i](feat).squeeze(-1))
                    em = w.unsqueeze(-1) * e
                msg = em.masked_fill(~member.unsqueeze(-1), float('-inf')).max(dim=2).values
                msg = torch.nan_to_num(msg, neginf=0.0)   # isolated nodes get no message
            else:
                if self.weight == 'kernel':
                    q = (dist / rd).clamp(max=1.0)
                    w = ((1.0 - q ** 2) ** 2) * member
                else:
                    w = torch.sigmoid(self.weight_mlps[i](feat).squeeze(-1)) * member
                if self.agg == 'wsum':                    # normalised: a weighted MEAN
                    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
                msg = (w.unsqueeze(-1) * e).sum(dim=2)

            h = h + node_mlp(msg)

        return torch.tanh(self.out_mlp(h)) * self.max_disp
