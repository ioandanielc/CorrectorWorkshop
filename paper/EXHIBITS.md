# Paper exhibits

Generated from `paper\results.csv` by `build_exhibits.py`. Every number
here is a row in that file — do not hand-edit.

**Seed variance** (measured, two runs of the identical recipe): viol_red +-1.4 points, |KG| +-10% relative.
Differences smaller than that are not separable from initialisation noise.

## E1 Architecture (matched params, identical loss)

| architecture | params | viol_red K5 | illegal% K5 | mean nn | \|KG\| K5 | vs best |
|---|---|---|---|---|---|---|
| model12_n49_noise0.6 | 350,594 | 82.9% | 0.69 | 0.1414 | 0.0216 | best |
| model12_n49_noise1.0 | 350,594 | 82.0% | 0.83 | 0.1412 | 0.0308 | +43% |
| dgcnn_n49_noise0.6 | 348,692 | 77.1% | 0.92 | 0.1403 | 0.0272 | +26% |
| dgcnn_n49_noise1.0 | 348,692 | 0.2% | 4.61 | 0.0919 | 0.3504 | +1521% |
| pointnet_n49_noise0.6 | 351,914 | 0.2% | 4.02 | 0.1003 | 0.2258 | +945% |
| pointnet_n49_noise1.0 | 351,914 | 0.1% | 4.62 | 0.0919 | 0.3502 | +1520% |
