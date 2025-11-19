import os
import h5py
import argparse
import json
import numpy as np

import dmg.demos
from dmg import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Name of the demonstration directory (inside dmg.demos.hdf5_root)",
    )
    parser.add_argument(
        "--play-dmp",
        action="store_true",
    )
    parser.add_argument(
        "--mimicgen",
        action="store_true",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Locate and open HDF5 file
    # ------------------------------------------------------------------
    if args.mimicgen:
        demo_path = os.path.join(dmg.demos.hdf5_root, "mimicgen", args.directory)
        if args.play_dmp:
            hdf5_path = os.path.join(demo_path, "dmp", "image.hdf5")
        else:
            hdf5_path = os.path.join(demo_path, "image.hdf5")
    else:
        demo_path = os.path.join(dmg.demos.hdf5_root, args.directory)
        if args.play_dmp:
            hdf5_path = os.path.join(demo_path, "dmp/image.hdf5")
        else:
            hdf5_path = os.path.join(demo_path, "image.hdf5")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "a") as f:
        if "data" not in f:
            raise KeyError("No 'data' group found in HDF5 file")

        # ------------------------------------------------------------------
        # Determine conversion direction from env_info
        # ------------------------------------------------------------------
        try:
            try:
                env_info = json.loads(f["data"].attrs["env_info"])
            except (KeyError, json.JSONDecodeError) as e:
                env_info = utils.from_env_args_to_env_info(json.loads(f["data"].attrs["env_args"]))
            control_delta = env_info["controller_configs"]["control_delta"]
        except Exception as e:
            raise RuntimeError(
                "Could not read 'env_info' or 'control_delta' from HDF5 attributes"
            ) from e

        convert_to = "absolute" if control_delta else "delta"
        print(f"\n🔧 Controller uses {'delta' if control_delta else 'absolute'} control.")
        print(f"→ Will convert actions from {('delta → absolute') if control_delta else ('absolute → delta')}\n")

        # ------------------------------------------------------------------
        # Process all demos
        # ------------------------------------------------------------------
        demo_names = list(f["data"].keys())
        if not demo_names:
            raise RuntimeError("No demos found under 'data/' group")

        print(f"Found {len(demo_names)} demos: {demo_names}\n")

        for demo_name in demo_names:
            demo_group = f["data"][demo_name]

            if "actions" not in demo_group:
                print(f"⚠️  Skipping {demo_name}: no 'actions' dataset found")
                continue

            # Delete existing converted dataset if present
            if "actions_abs" in demo_group:
                del demo_group["actions_abs"]
                print(f"🗑️  Deleted old 'actions_abs' in {demo_name}")

            # Load original actions
            actions = np.array(demo_group["actions"][()])
            print(f"→ Loaded 'actions' from {demo_name} (shape: {actions.shape})")

            # Perform conversion based on control_delta
            if control_delta:
                converted = utils.convert_delta_to_absolute(actions)
            else:
                converted = utils.convert_absolute_to_delta(actions)

            # Save converted dataset
            demo_group.create_dataset("actions_abs", data=converted)
            print(f"✅ Saved converted actions to 'actions_abs' for {demo_name}\n")

        f.flush()

    print("\n🎯 Conversion completed successfully!")
