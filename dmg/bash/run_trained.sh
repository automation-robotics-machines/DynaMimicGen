TASK_NAME=$1
TYPE=$2
CKPT=$3
HORIZON=$4
NUM=$5

python robomimic/robomimic/scripts/run_trained_agent.py \
--agent /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/models/model_epoch_${CKPT}.pth \
--n_rollouts 50 \
--horizon ${HORIZON} \
--seed 123 \
--video_path /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/model_epoch_${CKPT}_0.mp4 \
--succes_path /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/model_epoch_${CKPT}_0.json

python robomimic/robomimic/scripts/run_trained_agent.py \
--agent /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/models/model_epoch_${CKPT}.pth \
--n_rollouts 50 \
--horizon ${HORIZON} \
--seed 345 \
--video_path /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/model_epoch_${CKPT}_1.mp4 \
--succes_path /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/model_epoch_${CKPT}_1.json

python robomimic/robomimic/scripts/run_trained_agent.py \
--agent /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/models/model_epoch_${CKPT}.pth \
--n_rollouts 50 \
--horizon ${HORIZON} \
--seed 345678 \
--video_path /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/model_epoch_${CKPT}_2.mp4 \
--succes_path /mnt/arm_core/Dataset/DatasetVincenzo/${TASK_NAME}/diffusion_policy_trained_models/${TYPE}/test/${NUM}/model_epoch_${CKPT}_2.json