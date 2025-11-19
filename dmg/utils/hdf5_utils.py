import os
import h5py
from glob import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt

import robosuite

import dmg
import json

def gather_data_as_hdf5(directory, out_dir, env_info):#, dyn_params, gen_time):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point
    success_count = 0

    for demos_tried, ep_directory in enumerate(os.listdir(directory)):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            success_count+=1
            # print(f"Demonstration {demos_tried} is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            num_eps += 1

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        # else:
        #     print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    # grp.attrs["repository_version"] = dmg.__version__
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    
    
    success_rate = success_count/(demos_tried+1)
    success_dict = {
        "trajs successful": success_count,
        "trajs tried": demos_tried + 1,
        "success rate": success_rate
    }
    # grp.attrs["success_rate"] = success_rate
    # grp.attrs["gen_time"] = gen_time
    
    print(f"success rate: {success_rate}")
    # print("new generated dataset saved in "+ hdf5_path)
    with open(os.path.join(out_dir, "success.json"), "w") as f:
        json.dump(success_dict, f, indent=4)
    f.close()