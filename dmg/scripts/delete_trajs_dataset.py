import argparse
import os
import h5py
import numpy as np

def copy_attrs(src, dst):
    for key, val in src.attrs.items():
        dst.attrs[key] = val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, 
                        help="Relative path under dmg.demos.hdf5_root that contains the demo.hdf5 file")
    parser.add_argument(
        "--delete-demos", type=str, required=True,
        help="Comma-separated list of demo keys to replace, e.g., 'demo_0,demo_1,demo_5'"
    )
    parser.add_argument("--mimicgen", action="store_true")
    args = parser.parse_args()

    # Parse delete keys
    deleted_demo_keys = args.delete_demos.split(",")

    # Construct path
    nas_dir = "/mnt/arm_core/Dataset/DatasetVincenzo"
    if args.mimicgen:
        dir_path = os.path.join(nas_dir, "mimicgen", args.directory)
    else:
        dir_path = os.path.join(nas_dir, args.directory)
    dataset_path = os.path.join(dir_path, "dmp/image.hdf5")

    with h5py.File(dataset_path, "r+") as f:
        if "data" not in f:
            raise ValueError("'data' group not found in HDF5 file.")
        data_group = f["data"]

        # Ensure demo_0 exists
        if "demo_0" not in data_group:
            raise ValueError("demo_0 not found in the dataset; required for replacement.")

        for demo_key in deleted_demo_keys:
            # Delete existing demo group (if present)
            if demo_key in data_group:
                del data_group[demo_key]

            # Create and populate the replacement demo from demo_0
            new_demo = data_group.create_group(demo_key)
            for key, item in data_group["demo_0"].items():
                f.copy(f"data/demo_0/{key}", new_demo)
            copy_attrs(data_group["demo_0"], new_demo)

        total_demos = len(data_group.keys())

    print(f"✅ Replaced {len(deleted_demo_keys)} demos with demo_0 in {dataset_path}.")
    print(f"📦 Total number of demos now: {total_demos}")
