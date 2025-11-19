#!/bin/bash

TASK_NAME=$1
NUM_DEMOS=$2

python robomimic/robomimic/scripts/conversion/convert_robosuite.py --dataset /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/dmp/demo_${NUM_DEMOS}.hdf5 --add_absolute_actions
python robomimic/robomimic/scripts/dataset_states_to_obs.py --dataset /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/dmp/demo_${NUM_DEMOS}.hdf5 --output_name image${NUM_DEMOS}.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs