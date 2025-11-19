"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import os
import time
import shutil

import numpy as np

from robosuite.utils.mjcf_utils import save_sim_model
from robosuite.utils.transform_utils import mat2euler, quat2mat

from dmg.wrappers import CartesianControllerWrapper


class DMPsDataCollectionWrapper(CartesianControllerWrapper):
    def __init__(self, env, env_name, directory, rtb_robot, collect_freq=1, flush_freq=100, action_dim=None, action_spec=1.0):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env, rtb_robot, action_dim=action_dim, action_spec=action_spec)

        # Store env name for future use
        self.env_name = env_name
        
        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.dmp_num = 0
        self.states = []
        self.action_infos = []  # stores information about actions taken
        self.successful = False  # stores success state of demonstration

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            # print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

        # Env-Obj specific sensor names
        if "PickPlace" in self.env_name:
            self.object_pos_sensor_name = self.env.object_id_to_sensors[self.env.object_id][0]
            self.object_quat_sensor_name = self.env.object_id_to_sensors[self.env.object_id][1]
        elif self.env_name == "LiftObjects":
            self.object_pos_sensor_name = "object_pos"
            self.object_quat_sensor_name = "object_quat"
        elif self.env_name == "NutAssembly":
            if self.nut_id == 0:
                nut_type = "SquareNut"
            else:
                nut_type = "RoundNut"
            self.object_pos_sensor_name = nut_type + "_pos"
            self.object_quat_sensor_name = nut_type + "_quat"
        else:
            self.object_pos_sensor_name = "frame_pos"
            self.object_quat_sensor_name = "frame_quat"

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """
        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        # t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "dmp_"+str(self.dmp_num))
        assert not os.path.exists(self.ep_directory)
        # print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        # save initial state and action
        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            env=env_name,
        )
        self.states = []
        self.action_infos = []
        self.successful = False

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        self._start_new_episode()
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        # ob_dict, reward, terminated, info = ret
        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            info = {}
            info["actions"] = np.array(action)
            robot_base = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("robot0_base")].copy()
            gripper_position = self.env._get_observations()["robot0_eef_pos"]
            gripper_quat = self.env._get_observations()["robot0_eef_quat"]
            gripper_mat = quat2mat(gripper_quat).reshape(9, )
            gripper_base = np.append(gripper_position-robot_base, mat2euler(gripper_mat.reshape(3, 3), axes="rxyz"))
            info["robot0_eef_pose"] = gripper_base
            info["robot0_eef_pos"] = gripper_position
            info["robot0_eef_quat"] = gripper_quat
            info["robot0_eef_mat"] = gripper_mat
            info["robot0_gripper_qpos"] = self.env._get_observations()["robot0_gripper_qpos"]
            info["object_pos_world"] = self.env._get_observations()[self.object_pos_sensor_name]
            info["object_mat_world"] = quat2mat(self.env._get_observations()[self.object_quat_sensor_name]).reshape(9, )
            if self.env.use_camera_obs:
                for cn in self.env.camera_names:
                    img = self.env._get_observations()[cn+"_image"]
                    info[cn+"_image"] = img
            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True
        

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        # input(f"vis super action: {ret}")
        # print(f"self.successful: {self.successful}")
        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        if self.successful:
            self.dmp_num += 1
            if self.has_interaction:
                self._flush()
        else:
            shutil.rmtree(self.ep_directory)
            self.states = []
            self.action_infos = []
            self.successful = False
            self.has_interaction = False
        
        self.env.close()
