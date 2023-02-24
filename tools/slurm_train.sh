#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
#SRUN_ARGS=${SRUN_ARGS:-""}
SRUN_ARGS="--async -o $WORK_DIR/$JOB_NAME.log"
PY_ARGS=${@:5}
#PY_ARGS="--resume-from $WORK_DIR/latest.pth"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --quotatype=auto \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
