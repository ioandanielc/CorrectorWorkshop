import h5py

with h5py.File(rf"C:\Users\ioand\Downloads\phase_FluidName.h5part", "r") as f:
    keys = list(f.keys())
    print(f"Number of top-level groups: {len(keys)}")
    print(f"First few: {keys[:5]}")

    # Inspect one step's contents
    step0 = f[keys[0]]
    print(f"\nDatasets in {keys[0]}:")
    for name, ds in step0.items():
        print(f"  {name}: shape={ds.shape}, dtype={ds.dtype}")
