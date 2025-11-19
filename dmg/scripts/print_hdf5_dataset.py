import os
import json
import h5py
import argparse

import dmg.demos


def decode_and_print_attr(key, value, indent=2):
    """Decode and pretty-print an HDF5 attribute."""
    if key in {"model_file"}:
        return
    prefix = " " * indent
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
            try:
                parsed = json.loads(decoded)
                print(f"{prefix}{key}: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError:
                print(f"{prefix}{key}: {decoded}")
        except Exception:
            print(f"{prefix}{key}: (bytes) {value}")
    else:
        print(f"{prefix}{key}: {value}")


def print_hdf5_structure(name, obj):
    """Print structure with dataset info and attributes."""
    if isinstance(obj, h5py.Dataset):
        print(f"📄 Dataset: {name} | shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"📁 Group: {name}")
        if obj.attrs:
            print(f"  └─ Attributes:")
            for attr_key, attr_val in obj.attrs.items():
                decode_and_print_attr(attr_key, attr_val, indent=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the full content of an HDF5 demonstration file.")
    parser.add_argument("--directory", type=str,
                        help="Path to your demonstration directory that contains the demo.hdf5 file, e.g.: 'path_to_demos_dir/hdf5/YOUR_DEMONSTRATION'")
    parser.add_argument("--mimicgen", action="store_true")
    parser.add_argument("--play-dmp", action="store_true")
    args = parser.parse_args()

    # Build full directory path
    if args.mimicgen:
        dir_path = os.path.join(dmg.demos.hdf5_root, "mimicgen", args.directory)
    else:
        dir_path = os.path.join(dmg.demos.hdf5_root, args.directory)

    if args.play_dmp:
        dir_path = os.path.join(dir_path, "dmp")

    hdf5_files = [f for f in os.listdir(dir_path) if f.endswith(".hdf5")]

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in: {dir_path}")

    print("\nAvailable .hdf5 files:")
    for i, file in enumerate(hdf5_files, start=1):
        print(f"{i}: {file}")

    while True:
        ans = input(f"Select file [1-{len(hdf5_files)}] (default=1): ").strip()
        if ans == "":
            selection = 1
            break
        elif ans.isdigit() and 1 <= int(ans) <= len(hdf5_files):
            selection = int(ans)
            break
        else:
            print("Invalid input. Please enter a valid number or press Enter for default.")

    filename = hdf5_files[selection - 1]
    path = os.path.join(dir_path, filename)

    print(f"\n✅ Selected file: {filename}")
    print(f"📂 Full path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found at: {path}")

    print(f"\n🔍 Inspecting HDF5 file: {path}\n")

    with h5py.File(path, "r") as f:
        print("=== HDF5 File Structure ===")
        f.visititems(print_hdf5_structure)

        print("\n=== Root-level Attributes ===")
        if f.attrs:
            for key, val in f.attrs.items():
                decode_and_print_attr(key, val, indent=2)
        else:
            print("  (none)")

        # Explicitly handle 'data' group attributes (env_args, etc.)
        if "data" in f:
            print("\n=== 'data' Group Attributes ===")
            for key, val in f["data"].attrs.items():
                decode_and_print_attr(key, val, indent=2)

            print("\n=== Demonstration Episodes ===")
            for demo_key in f["data"]:
                demo = f["data"][demo_key]
                print(f"\n▶ Episode: {demo_key}")
                if demo.attrs:
                    print("  └─ Attributes:")
                    for attr_key, attr_val in demo.attrs.items():
                        decode_and_print_attr(attr_key, attr_val, indent=6)
                else:
                    print("  (no attributes)")
        else:
            print("\n(no 'data' group found in this file)")
