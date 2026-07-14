"""
src/inference/correctors/common/scaling.py
-------------------
Coordinate rescaling so the model always operates at its training rd.

The corrector was trained on clouds with rd = rd_train.
SPH data has rd = rd_test (different scale).

Procedure for one tile:
  1. Scale coords:    pts_scaled = pts * scale          (scale = rd_train / rd_test)
  2. make_invariant(pts_scaled) -> invariant representation
  3. model(invariant, rd=rd_train) -> displacement in invariant space
  4. Revert invariant -> corrected_scaled
  5. Unscale displacement: disp_original = (corrected_scaled - pts_scaled) / scale

After this, displacements are in the original coordinate system and can be
directly added to the original (unscaled) positions.
"""


def compute_scale(rd_train: float, rd_test: float) -> float:
    """scale = rd_train / rd_test  (multiply coords before inference)."""
    return rd_train / rd_test
