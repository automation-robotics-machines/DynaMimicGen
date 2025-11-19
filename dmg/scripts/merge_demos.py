import os
import numpy as np
import h5py
import argparse
import dmg.demos

def copy_attrs(src, dst):
    """Copy attributes from one HDF5 object to another."""
    for key, val in src.attrs.items():
        dst.attrs[key] = val

def copy_non_demo_data(src_file, dst_file, prefix=""):
    """Copy all groups and datasets except 'data' from src to dst with optional prefix."""
    for key in src_file.keys():
        if key != "data":
            src_file.copy(key, dst_file, name=f"{prefix}{key}")
            copy_attrs(src_file[key], dst_file[f"{prefix}{key}"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, 
                        help="Environment name for the merged dataset, e.g. 'Square_D0' or 'MugCleanup_D0'")
    args = parser.parse_args()

    # Input HDF5 paths
    demo1_path = os.path.join(dmg.demos.hdf5_root, "demos/2Demo/Square/Square_D1First/demo.hdf5")
    demo2_path = os.path.join(dmg.demos.hdf5_root, "demos/2Demo/Square/Square_D1Second/demo.hdf5")

    # Output HDF5 path
    out_dir = os.path.join(dmg.demos.hdf5_root, f"demos/2Demo/{args.directory}")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "demo.hdf5")

    with h5py.File(demo1_path, "r") as f1, h5py.File(demo2_path, "r") as f2, h5py.File(output_path, "w") as f_out:
        # Get the first demo key in both files
        demo1_key = sorted(list(f1["data"].keys()), key=lambda x: int(x.split("_")[1]))[0]
        demo2_key = sorted(list(f2["data"].keys()), key=lambda x: int(x.split("_")[1]))[0]

        # Create the 'data' group
        data_group = f_out.create_group("data")

        # Copy demo_1 from f1
        f1.copy(f"data/{demo1_key}", data_group, name="demo_1")
        copy_attrs(f1["data"][demo1_key], data_group["demo_1"])

        # Copy demo_2 from f2
        f2.copy(f"data/{demo2_key}", data_group, name="demo_2")
        copy_attrs(f2["data"][demo2_key], data_group["demo_2"])

        # Copy metadata and other dataset fields (with prefixes to avoid conflict)
        copy_non_demo_data(f1, f_out, prefix="file1_")
        copy_non_demo_data(f2, f_out, prefix="file2_")

        # Optional: copy attributes of 'data' group if needed
        copy_attrs(f1["data"], f_out["data"])

    print(f"✅ Merged demos saved to: {output_path}")
