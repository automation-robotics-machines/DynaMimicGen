import argparse
import json
import os
import h5py

import dmg.demos


def build_demo_path(args) -> str:
    """Construct the demo path based on provided arguments."""
    base_path = os.path.join(
        dmg.demos.hdf5_root,
        "mimicgen" if args.mimicgen else "",
        args.directory,
    )
    if args.play_dmp:
        base_path = os.path.join(base_path, "dmp")
    return base_path


def list_hdf5_files(path: str) -> list:
    """Return a sorted list of .hdf5 files in the given directory."""
    files = sorted(f for f in os.listdir(path) if f.endswith(".hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {path}")
    return files


def choose_file(files: list) -> str:
    """Prompt user to select a file from the list of HDF5 files."""
    print("\n📂 Available HDF5 files:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    while True:
        try:
            choice = int(input("\nEnter the number of the file you want to modify: "))
            if 0 <= choice < len(files):
                return files[choice]
            print(f"❌ Invalid choice. Please enter a number between 0 and {len(files) - 1}.")
        except ValueError:
            print("❌ Please enter a valid number.")


def load_env_args(h5_file: h5py.File) -> dict:
    """Load and decode env_args from the HDF5 file."""
    raw = h5_file["data"].attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def toggle_control_delta(env_args: dict) -> dict:
    """Toggle the 'control_delta' parameter with user confirmation."""
    current = env_args["env_kwargs"]["controller_configs"]["control_delta"]
    ans = input(f"\nCurrent 'control_delta' is {current}. Change it? (y/n): ").strip().lower()

    if ans not in {"y", "yes"}:
        print("⚠️  No changes made.")
        return env_args

    env_args["env_kwargs"]["controller_configs"]["control_delta"] = not current
    print(f"✅ 'control_delta' changed to {env_args['env_kwargs']['controller_configs']['control_delta']}.")
    return env_args


def main():
    parser = argparse.ArgumentParser(description="Modify env_args inside an HDF5 demo file.")
    parser.add_argument("--directory", type=str, required=True,
                        help="Path to your demonstration directory that contains the demo.hdf5 file")
    parser.add_argument("--mimicgen", action="store_true", help="Look inside the mimicgen subfolder")
    parser.add_argument("--play-dmp", action="store_true", help="Look inside the dmp subfolder")
    args = parser.parse_args()

    demo_path = build_demo_path(args)
    files = list_hdf5_files(demo_path)
    file_name = choose_file(files)
    hdf5_file = os.path.join(demo_path, file_name)

    print(f"\n✅ Selected file: {hdf5_file}")

    with h5py.File(hdf5_file, "r+") as f:
        env_args = load_env_args(f)
        updated_env_args = toggle_control_delta(env_args)
        # Save only if changed
        f["data"].attrs["env_args"] = json.dumps(updated_env_args).encode("utf-8")

    print(f"\n📦 File modified: {hdf5_file}")


if __name__ == "__main__":
    main()
