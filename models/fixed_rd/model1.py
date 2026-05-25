import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CorrectorModel(nn.Module):
    def __init__(self, model_config: dict, input_dim: int, initialization: str):
        super().__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.num_attention_modules = model_config['num_attention_modules']
        self.use_batch_norm = model_config['batch_norm']
        self.output_dim = input_dim

        activation_cls = getattr(nn, model_config['activation'])

        def make_mlp(in_dim, out_dim):
            layers = [nn.Linear(in_dim, out_dim)]
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation_cls())
            return nn.ModuleList(layers)

        self.input_mlp = make_mlp(input_dim, self.hidden_dim)

        self.attention_modules = nn.ModuleList([
            nn.ModuleList([
                make_mlp(self.hidden_dim, self.hidden_dim),   # query
                make_mlp(self.hidden_dim, self.hidden_dim),   # key
                make_mlp(self.hidden_dim, self.hidden_dim),   # value
                make_mlp(self.hidden_dim * 2, self.hidden_dim),  # concat
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

    def _apply_module(self, x, module):
        for layer in module:
            if isinstance(layer, nn.BatchNorm1d):
                x = x.permute(0, 2, 1)
                x = layer(x)
                x = x.permute(0, 2, 1)
            else:
                x = layer(x)
        return x

    def forward(self, x, return_attention_maps=False):
        # x: (B, N, dim)
        attention_maps = []

        x = x - x.mean(dim=1, keepdim=True)
        x = self._apply_module(x, self.input_mlp)

        for query_mlp, key_mlp, value_mlp, concat_mlp in self.attention_modules:
            q = self._apply_module(x, query_mlp)
            k = self._apply_module(x, key_mlp)
            v = self._apply_module(x, value_mlp)

            weights = F.softmax(
                torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5),
                dim=-1
            )
            attention_maps.append(weights.detach().clone())

            attended = torch.bmm(weights, v)
            x = self._apply_module(torch.cat([x, attended], dim=-1), concat_mlp)

        x = self.output_layer(x)

        if return_attention_maps:
            return x, torch.stack(attention_maps)
        return x
