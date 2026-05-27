"""
model5: model4 + LayerNorm (replacing BatchNorm).

Changes vs model4:
- LayerNorm(hidden_dim) instead of BatchNorm1d — operates directly on (B, N, C),
  no permute needed. MLPs are nn.Sequential instead of nn.ModuleList, so
  _apply_module is gone entirely.
- Everything else is identical to model4: no centroid subtraction, concatenative
  skip (hidden_dim * 2 → hidden_dim), linear output, no additive residuals.

Config key: norm (options: 'layer', 'none') — replaces batch_norm bool.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CorrectorModel(nn.Module):
    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.num_attention_modules = model_config['num_attention_modules']
        self.norm_type = model_config['norm']   # 'layer' or 'none'
        self.output_dim = input_dim

        activation_cls = getattr(nn, model_config['activation'])

        def make_mlp(in_dim, out_dim):
            layers = [nn.Linear(in_dim, out_dim)]
            if self.norm_type == 'layer':
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation_cls())
            return nn.Sequential(*layers)

        self.input_mlp = make_mlp(input_dim, self.hidden_dim)

        self.attention_modules = nn.ModuleList([
            nn.ModuleList([
                make_mlp(self.hidden_dim, self.hidden_dim),          # query
                make_mlp(self.hidden_dim, self.hidden_dim),          # key
                make_mlp(self.hidden_dim, self.hidden_dim),          # value
                make_mlp(self.hidden_dim * 2, self.hidden_dim),      # concat: x + attended
            ])
            for _ in range(self.num_attention_modules)
        ])

        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
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

    def forward(self, x, return_attention_maps=False):
        # x: (B, N, dim) — centroid already removed by DataProcessor
        # LayerNorm operates on last dim directly, no permute needed
        attention_maps = []

        x = self.input_mlp(x)

        for query_mlp, key_mlp, value_mlp, concat_mlp in self.attention_modules:
            q = query_mlp(x)
            k = key_mlp(x)
            v = value_mlp(x)

            weights = F.softmax(
                torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5),
                dim=-1
            )
            attention_maps.append(weights.detach().clone())

            attended = torch.bmm(weights, v)
            x = concat_mlp(torch.cat([x, attended], dim=-1))

        x = self.output_layer(x)

        if return_attention_maps:
            return x, torch.stack(attention_maps)
        return x
