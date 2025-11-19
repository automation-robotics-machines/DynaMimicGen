import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.objects import CanObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

# from suite.models.arenas import BoxArena
from robosuite.models.arenas import EmptyArena

from dmg.models.objects import TargetObject

class Void(SingleArmEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=False,
        use_object_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        reward_shaping=True,
        reward_scale=0.1,
        # init_qpos = np.array([.0,.0,.0,.0,.0,.0])
    ):
        self.mujoco_arena = EmptyArena()
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            # init_qpos=init_qpos,
        )
        self.current_step = 0
        self.reward_shaping = reward_shaping
        self.reward_scale   = reward_scale
        self.use_object_obs = use_object_obs

        
    def reward(self, action=None):
        reward = 0.1
        return reward

    def _load_model(self):
        super()._load_model()

        self.robots[0].robot_model.set_base_xpos((0.0, 0.0, 0))
        
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])
                
        target = TargetObject(name="target")
        
            
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=target,
        )
        
    def _setup_references(self):
        super()._setup_references()

    def _setup_observables(self):
        observables = super()._setup_observables()
        return observables
        
        # modality = "object"
        pf =  self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"
        @sensor(modality=modality)
        def force_sensor(obs_cache):
            ret = self.robots[0].get_sensor_measurement("gripper0_force_ee")
            return ret
        @sensor(modality=modality)
        def torque_sensor(obs_cache):
            ret = self.robots[0].get_sensor_measurement("gripper0_torque_ee")
            return ret

        sensors = [force_sensor,torque_sensor]
        names   = ["ee_force","ee_torque"]
        enableds = [True,True]
        actives = [True,True]

        for name, s, enabled, active in zip(names, sensors, enableds, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                enabled=enabled,
                active=active,
            )
        
        ret = OrderedDict()
        
        ret["ee_pos"] = observables["robot0_eef_pos"]
        ret["ee_quat"] = observables["robot0_eef_quat"]
        ret["ee_force"] = observables["ee_force"]
        ret["ee_torque"] = observables["ee_torque"]
        
        return ret


    def _reset_internal(self):
        
        super()._reset_internal()


    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
    
    def _check_success(self):
        """
        The task is considered successfull when the horizon is reached.

        Returns:
            bool: True if self.current_step == self.horizon
        """
        if self.current_step < self.horizon:
            self.current_step += 1

        return self.current_step == self.horizon