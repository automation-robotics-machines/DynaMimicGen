#!/bin/bash

TASK_NAME=$1

python robomimic/robomimic/scripts/conversion/convert_robosuite.py --dataset /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/demo.hdf5 --add_delta_actions
python dmg/scripts/change_delta.py --directory ${TASK_NAME} # --play-dmp
python dmg/scripts/playback_delta_actions.py --directory ${TASK_NAME} --use-actions # --play-dmp