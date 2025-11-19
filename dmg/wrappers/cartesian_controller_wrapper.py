"""
This file implements a wrapper for facilitating compatibility with OpenAI gym and standard Robosuite controollers.
This wrapper re-implements the step function to allow cartesian command actions and joint position control.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env

from robosuite.wrappers import GymWrapper

from spatialmath import SE3

class CartesianControllerWrapper(GymWrapper):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, rtb_robot, action_is_delta=False, admittance=False, action_dim=3, action_spec=0.1, keys=None):
        # Run super method
        super().__init__(env=env)
        self.action_dimensions=action_dim
        
        # if len(self.action_spec) == 1:
        self.action_specifications = (np.array([-action_spec]*self.action_dimensions),np.array([action_spec]*self.action_dimensions))
        # elif len(self.action_spec) != len(self.action_dimensions):
        #     raise ValueError("Action specification must be a scalar or a list of length equal to the action dimensions")
        # else:
            # self.action_specifications=(np.array(-action_spec),np.array(action_spec))
        
        low, high = self.action_specifications
        self.action_space = spaces.Box(low, high)
        self.joints = self.robots[0]._ref_joint_pos_indexes
        self.rtb_robot = rtb_robot
        
        self.delta_goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.m = 1.0
        self.k = 0.0
        
        # self.c = 0.9*2*np.sqrt(self.k*self.m)
        self.c = 20.0
        
        self.xd = 0
        self.x0 = 0
        self.first_cycle=True
        self.admittance = admittance
        self.dt = 1/self.control_freq
        
        self.action_is_delta = action_is_delta
        
    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        # input(f"contr action: {action}")
        joint_pos = self.sim.data.qpos[self.joints]
        if self.action_is_delta:
            
            T_fk = self.rtb_robot.fkine(joint_pos)
            
            if self.first_cycle:
                self.x0 = T_fk.t[1]
                self.first_cycle = False
            
            if self.admittance:
                # m xdd + c xd + k x = u --> xdd = (u - c xd - k x)/m
                # ADMITTANCE
                eef_body_name = "robot0_right_hand" # TODO:: get it from robot[0]        
                linear_velocity      = self.sim.data.get_body_xvelp(eef_body_name)
                angular_velocity     = self.sim.data.get_body_xvelr(eef_body_name)
                xdd = (action[1] + self.c*linear_velocity[1] + self.k*(T_fk.t[1]-self.x0))/self.m
                self.xd += xdd*self.dt
                self.delta_x = self.xd*self.dt 
                self.delta_goal[1] = self.xd
                
                # ADMITTANCE
            else:
                for i in range(self.action_dimensions):
                    self.delta_goal[i] = action[i]
            
            T_trans = SE3(self.delta_goal[0], self.delta_goal[1], self.delta_goal[2])        
            T_rot = SE3.Rx(self.delta_goal[3], unit="rad") * SE3.Ry(self.delta_goal[4], unit="rad") * SE3.Rz(self.delta_goal[5], unit="rad")
            T_rel = T_trans * T_rot
            T_goal = T_fk * T_rel
            
        else:
            T_goal = SE3(action[0], action[1], action[2]) * SE3.Rx(action[3], unit="rad") * SE3.Ry(action[4], unit="rad") * SE3.Rz(action[5], unit="rad")
        
        joint_cmd, _, _, _, _ = self.rtb_robot.ik_LM(T_goal, q0=joint_pos)
        # joint_cmd, _, _, _, _ = self.rtb_robot.ik_NR(T_goal, q0=joint_pos)
        
        delta_sol = joint_cmd - joint_pos

        action = np.append(delta_sol, action[-1])
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    @property
    def action_dim(self):
        """
        ovverrides the normal environment action_dim to fit with the Cartesian desired DOFs

        Returns:
            int: Action space dimension
        """
        return self.action_dimensions
    
    @property
    def action_spec(self):
        """
        overrides the normal environment action_spec to fit with the Cartesian desired DOFs

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        return self.action_specifications