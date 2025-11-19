import argparse
import os
import random
import h5py
import numpy as np
from tqdm import tqdm

def copy_attrs(src, dst):
    for key, val in src.attrs.items():
        dst.attrs[key] = val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, 
                        help="Relative path under dmg.demos.hdf5_root that contains the demo.hdf5 file")
    parser.add_argument("--num-demos", type=int, default=200,
                        help="Number of demos to keep (default: 200)")
    parser.add_argument("--mimicgen", action="store_true")
    args = parser.parse_args()

    # Construct source path
    nas_dir = "/mnt/arm_core/Dataset/DatasetVincenzo"
    if args.mimicgen:
        dir_path = os.path.join(nas_dir, "mimicgen", args.directory)
    else:
        dir_path = os.path.join(nas_dir, args.directory)
    dataset_path = os.path.join(dir_path, "dmp/demo.hdf5")

    # Construct output path
    output_path = os.path.join(dir_path, "dmp", f"demo_{args.num_demos}.hdf5")

    with h5py.File(dataset_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # Copy non-data groups (e.g., metadata, env_args, etc.)
        for key in f_in.keys():
            if key != "data":
                f_in.copy(key, f_out)
                copy_attrs(f_in[key], f_out[key])

        # Create the 'data' group
        f_out.create_group("data")

        # Get sorted demo keys and filter first N
        all_demos = sorted(list(f_in["data"].keys()), key=lambda x: int(x.split("_")[1]))
        selected_demos = all_demos[:args.num_demos]

        for demo_key in tqdm(selected_demos, desc=f"Copying demos {args.num_demos}", unit="demo"):
            f_in.copy(f_in[f"data/{demo_key}"], f_out[f"data"], name=demo_key)
            copy_attrs(f_in[f"data/{demo_key}"], f_out[f"data/{demo_key}"])

        # Copy attributes of the 'data' group itself
        copy_attrs(f_in["data"], f_out["data"])

    print(f"✅ Saved {len(selected_demos)} demos to: {output_path}")
