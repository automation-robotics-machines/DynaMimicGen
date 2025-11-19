import argparse
import os
import h5py
import numpy as np
import json

def copy_attrs(src, dst):
    for key, val in src.attrs.items():
        dst.attrs[key] = val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=str,
        help="Relative path under dmg.demos.hdf5_root that contains the demo.hdf5 file"
    )
    parser.add_argument(
        "--keep-demos", type=str, required=True,
        help="Comma-separated list of demo keys to keep, e.g., 'demo_0,demo_1,demo_5'"
    )
    args = parser.parse_args()

    # Construct source path
    nas_dir = "/mnt/arm_core/Dataset/DatasetVincenzo"
    dir_path = os.path.join(nas_dir, args.directory)
    dataset_path = os.path.join(dir_path, "image.hdf5")

    # Convert input string to list of demo keys
    selected_demos = [demo.strip() for demo in args.keep_demos.split(",")]

    # Construct output path
    output_path = os.path.join(dir_path, "image_filtered.hdf5")

    with h5py.File(dataset_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # ✅ Copy file-level (root) attributes
        copy_attrs(f_in, f_out)

        # Copy non-data groups (e.g., metadata, env_args, etc.)
        for key in f_in.keys():
            if key != "data":
                f_in.copy(key, f_out)
                copy_attrs(f_in[key], f_out[key])

        # Create the 'data' group
        f_out.create_group("data")

        # Validate demo keys
        all_demos = sorted(list(f_in["data"].keys()))
        for demo_key in selected_demos:
            if demo_key not in all_demos:
                raise ValueError(f"❌ Demo '{demo_key}' not found in dataset.")

        # Copy selected demos
        for demo_num, demo_key in enumerate(selected_demos):
            source_key = f"demo_{demo_num}"
            f_in.copy(f_in[f"data/{demo_key}"], f_out["data"], name=source_key)
            copy_attrs(f_in[f"data/{demo_key}"], f_out["data"][source_key])

        # Copy attributes of the 'data' group itself
        copy_attrs(f_in["data"], f_out["data"])

    print(f"✅ Saved {len(selected_demos)} selected demos to: {output_path}")
