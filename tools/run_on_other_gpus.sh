#!/usr/bin/env bash

set -x

PARTITION=sensedeep
JOB_NAME=hello
CONFIG_FILE=configs/cooperative-pretraining-baselines-v100/cooperative_pretraining_unet_fusion_encoder_no_grad_fileter_ground_points_1_voxel_contrastive_2_shape_context_prediction_80epochs.py
WORK_DIR=./work_dirs_cooperative_pretraining/cooperative_pretraining_unet_fusion_encoder_no_grad_fileter_ground_points_1_voxel_contrastive_2_shape_context_prediction_80epochs
GPUS=8
GPUS_PER_NODE=8
SRUN_ARGS="--async -o $WORK_DIR/$JOB_NAME.log"
PY_ARGS="--resume-from $WORK_DIR/latest.pth"
#QUOTATYPE=auto

GPUS=${GPUS} . ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${PY_ARGS}