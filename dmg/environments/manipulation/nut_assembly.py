# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
from six import with_metaclass

import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.nut_assembly import NutAssembly, NutAssemblySquare
from robosuite.models.arenas import PegsArena
from robosuite.models.objects import SquareNutObject, RoundNutObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, find_elements
from robosuite.utils import RandomizationError

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from mimicgen.envs.robosuite.nut_assembly import Square_D0

class NutAssemblyMod(NutAssembly):
    """
    Specifies a different placement initializer for the pegs where it is initialized
    with a broader x-range and broader y-range.
    """
    def _get_initial_placement_bounds(self):
        return dict(
            square_nut=dict(
                x=(-0.2, -0.2),
                y=(0.3, 0.3),
                z_rot=(0),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            round_nut=dict(
                x=(-0.2, -0.2),
                y=(-0.3, -0.3),
                z_rot=(0),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg1=dict(
                x=(0.1, 0.1),
                y=(0.1, 0.1),
                z_rot=(0., 0.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg2=dict(
                x=(0.1, 0.1),
                y=(-0.1, -0.1),
                z_rot=(0., 0.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = np.random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation 
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg_bounds = self._get_initial_placement_bounds()["peg1"]

        sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        sample_z_rot = np.random.uniform(low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1])
        peg1_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        peg1_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        peg1_xml_quat = np.array([np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)])

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg1_xml.set("quat", array_to_string(peg1_xml_quat))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.nuts,
        )

    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            peg1_id = self.sim.model.body_name2id("peg1")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[peg1_id])

            name = "peg1_pos"
            observables[name] = Observable(
                name=name,
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )

        return observables