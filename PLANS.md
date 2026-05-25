# Future Plans

## Model Architecture

### Pending changes (in order of priority)

1. **Additive residuals**
   - Replace the current concatenative skip (`cat([x, attended])`) with a standard additive residual
   - Each attention block becomes: `x = concat_mlp(cat([x, attended])) + x`
   - Requires `concat_mlp` input to be `hidden_dim * 2 → hidden_dim` (already the case)

2. **LayerNorm instead of BatchNorm**
   - Replace `BatchNorm1d` + permute hack with `nn.LayerNorm(hidden_dim)`
   - Applied directly on `(B, N, C)` — no permutation needed
   - Update `batch_norm` key in model config to `norm: layer` with options `none / layer`

3. **`tanh` output + `max_displacement` scaling**
   - Add `tanh` activation on the output layer
   - Scale by `max_displacement` scalar: `output = max_displacement * tanh(self.output_layer(x))`
   - Add `max_displacement` to `model_config`
