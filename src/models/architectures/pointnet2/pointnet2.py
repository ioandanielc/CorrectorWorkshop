"""
PointNet++ corrector — the hierarchical baseline.

Set abstraction down, feature propagation up, after Qi et al. (2017). This is the
only model in the suite that grows its receptive field by *pooling* rather than by
iterating local message passing: farthest-point sampling picks centroids, a ball
query groups neighbours around each, a shared mini-PointNet max-pools the group into
a centroid feature, and the coarse features are then interpolated back onto the
original points with skip connections.

It therefore tests a mechanism the others do not have — whether a coarse-to-fine
hierarchy can substitute for message-passing depth. The prediction cutting against it
is that feature propagation *interpolates*, which smooths, while the task needs
per-point precision well below rd.

Periodicity is handled throughout: sampling, grouping and interpolation all use
minimum-image distances, so it is not handicapped relative to the others.

Config keys: hidden_dim, npoint_frac, radius_spacings, nsample, norm, activation,
max_displacement, cutoff_rd.
"""
import torch
import torch.nn as nn
import torch.nn.init as init


def _rel(a, b, box):
    """Minimum-image a[:, :, None] - b[:, None, :]. a: (B, M, D), b: (B, S, D)."""
    r = a.unsqueeze(2) - b.unsqueeze(1)          # (B, M, S, D)
    if box is not None:
        r = r - box * torch.round(r / box)
    return r


def farthest_point_sample(x, n_samples, box):
    """(B, n_samples) indices, greedily maximising minimum distance on the torus."""
    B, N, D = x.shape
    idx = torch.zeros(B, n_samples, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), float('inf'), device=x.device)
    far = torch.zeros(B, dtype=torch.long, device=x.device)
    ar = torch.arange(B, device=x.device)
    for i in range(n_samples):
        idx[:, i] = far
        c = x[ar, far].unsqueeze(1)               # (B, 1, D)
        d = x - c
        if box is not None:
            d = d - box * torch.round(d / box)
        dist = torch.minimum(dist, (d ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


def ball_query(x, centroids, radius, nsample, box):
    """(B, S, nsample) neighbour indices; out-of-radius slots fall back to the nearest."""
    d2 = (_rel(centroids, x, box) ** 2).sum(-1)   # (B, S, N)
    near = d2.argsort(dim=-1)[..., :nsample]      # (B, S, nsample) nearest first
    taken = d2.gather(2, near)
    outside = taken > radius ** 2
    return torch.where(outside, near[..., :1].expand_as(near), near)


class CorrectorModel(nn.Module):
    uses_rd  = True
    uses_box = True

    def __init__(self, model_config, input_dim, initialization):
        super().__init__()
        H = model_config['hidden_dim']
        self.hidden_dim = H
        self.max_disp   = model_config.get('max_displacement', 0.06)
        self.npoint_frac = model_config.get('npoint_frac', [4, 16])       # N/4, N/16
        self.radius_sp   = model_config.get('radius_spacings', [2.0, 4.0])
        self.nsample     = model_config.get('nsample', [12, 12])
        self.dim = input_dim

        act    = getattr(nn, model_config['activation'])
        use_ln = model_config.get('norm') == 'layer'

        def mlp(d_in, d_out):
            layers = [nn.Linear(d_in, H)]
            if use_ln:
                layers.append(nn.LayerNorm(H))
            layers += [act(), nn.Linear(H, d_out)]
            return nn.Sequential(*layers)

        # set abstraction: group features are [neighbour feature, rel position]
        self.sa1 = mlp(input_dim, H)              # level 1 sees geometry only
        self.sa2 = mlp(H + input_dim, H)
        # feature propagation: [interpolated coarse feature, skip feature]
        self.fp2 = mlp(2 * H, H)
        self.fp1 = mlp(H + H, H)
        self.head = nn.Sequential(nn.Linear(H, H), act(), nn.Linear(H, input_dim))
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

    def _abstract(self, xyz, feat, npoint, radius, nsample, net, box):
        """Sample centroids, group, shared MLP, max-pool -> (centroid xyz, features)."""
        B, N, D = xyz.shape
        npoint = max(1, min(npoint, N))
        idx = farthest_point_sample(xyz, npoint, box)
        ar = torch.arange(B, device=xyz.device).unsqueeze(-1)
        new_xyz = xyz[ar, idx]                                     # (B, S, D)

        nb = ball_query(xyz, new_xyz, radius, min(nsample, N), box)   # (B, S, k)
        rel = xyz[ar.unsqueeze(-1), nb] - new_xyz.unsqueeze(2)        # (B, S, k, D)
        if box is not None:
            rel = rel - box * torch.round(rel / box)

        grouped = rel if feat is None else torch.cat(
            [feat[ar.unsqueeze(-1), nb], rel], dim=-1)
        return new_xyz, net(grouped).max(dim=2).values              # max over the group

    def _propagate(self, xyz, sub_xyz, feat, skip, net, box):
        """Inverse-distance-weighted 3-NN interpolation back onto xyz, then MLP."""
        d2 = (_rel(xyz, sub_xyz, box) ** 2).sum(-1)                  # (B, N, S)
        k = min(3, sub_xyz.shape[1])
        d2, idx = d2.topk(k, dim=-1, largest=False)
        w = 1.0 / (d2.sqrt() + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)                          # (B, N, k)

        ar = torch.arange(xyz.shape[0], device=xyz.device).unsqueeze(-1)
        interp = (feat[ar.unsqueeze(-1), idx] * w.unsqueeze(-1)).sum(dim=2)
        return net(torch.cat([interp, skip], dim=-1))

    def forward(self, x, rd=None, box=None):
        if rd is None:
            raise ValueError("pointnet2 requires rd (sets the ball-query radii)")
        B, N, D = x.shape
        spacing = float(rd) / 2.0        # cutoff_rd is 2 spacings by construction
        r1, r2 = (s * spacing for s in self.radius_sp)

        xyz1, f1 = self._abstract(x, None, N // self.npoint_frac[0], r1,
                                  self.nsample[0], self.sa1, box)
        xyz2, f2 = self._abstract(xyz1, f1, N // self.npoint_frac[1], r2,
                                  self.nsample[1], self.sa2, box)

        f1 = self._propagate(xyz1, xyz2, f2, f1, self.fp2, box)
        f0 = self._propagate(x, xyz1, f1, x.new_zeros(B, N, self.hidden_dim),
                             self.fp1, box)
        return torch.tanh(self.head(f0)) * self.max_disp
