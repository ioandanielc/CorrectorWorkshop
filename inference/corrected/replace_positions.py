"""
Replace coords_0 / coords_1 in an H5Part file using positions from a
custom text file of the form:

    # t=0
    x0 y0
    x1 y1
    ...
    # t=1
    x0 y0
    ...

Only x/y (coords_0, coords_1) are overwritten. coords_2 (z) and all
other datasets (Velocity, Pressure, Density, ...) are left untouched.

Usage:
    python replace_positions.py <positions.txt> <input.h5part> <output.h5part>
"""

import sys
import re
import numpy as np
import h5py


def parse_positions_file(path):
    """
    Parses the custom text format into a dict: {step_index: np.ndarray of shape (N, 2)}
    """
    positions = {}
    current_step = None
    current_rows = []

    step_re = re.compile(r"^#\s*t\s*=\s*(\d+)\s*$")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                m = step_re.match(line)
                if m:
                    # flush previous step
                    if current_step is not None:
                        positions[current_step] = np.array(current_rows, dtype=np.float32)
                    current_step = int(m.group(1))
                    current_rows = []
                # other comment lines (header metadata) are ignored
                continue

            # data line: "x y"
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Unexpected line format: {line!r}")
            x, y = float(parts[0]), float(parts[1])
            current_rows.append((x, y))

        # flush last step
        if current_step is not None:
            positions[current_step] = np.array(current_rows, dtype=np.float32)

    return positions


def replace_positions(positions_path, input_h5part, output_h5part):
    print(f"Parsing {positions_path} ...")
    positions = parse_positions_file(positions_path)
    n_steps_parsed = len(positions)
    print(f"Parsed {n_steps_parsed} timesteps from positions file.")

    # sanity check on particle count consistency
    counts = {t: arr.shape[0] for t, arr in positions.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        print("WARNING: inconsistent particle counts across steps:", unique_counts)
    else:
        print(f"Particles per step: {unique_counts.pop()}")

    print(f"Copying {input_h5part} -> {output_h5part} ...")
    import shutil
    shutil.copyfile(input_h5part, output_h5part)

    print("Writing new positions ...")
    with h5py.File(output_h5part, "r+") as f:
        step_keys = list(f.keys())
        print(f"Found {len(step_keys)} steps in H5Part file.")

        n_written = 0
        n_missing_in_file = 0
        n_skipped_no_data = 0

        for step_idx, new_xy in positions.items():
            group_name = f"Step#{step_idx}"
            if group_name not in f:
                n_missing_in_file += 1
                continue

            grp = f[group_name]
            n_particles_h5 = grp["coords_0"].shape[0]

            if new_xy.shape[0] != n_particles_h5:
                print(f"  SKIP {group_name}: h5 has {n_particles_h5} particles, "
                      f"positions file has {new_xy.shape[0]}")
                n_skipped_no_data += 1
                continue

            grp["coords_0"][:] = new_xy[:, 0]
            grp["coords_1"][:] = new_xy[:, 1]
            # coords_2 left untouched
            n_written += 1

        print(f"\nDone.")
        print(f"  Steps written:            {n_written}")
        print(f"  Steps in positions file but missing from h5: {n_missing_in_file}")
        print(f"  Steps skipped (particle count mismatch):     {n_skipped_no_data}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    positions_path = sys.argv[1]
    input_h5part = sys.argv[2]
    output_h5part = sys.argv[3]

    replace_positions(positions_path, input_h5part, output_h5part)
