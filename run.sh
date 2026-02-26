# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

#!/usr/bin/env bash
#
# usage:
# ./run.sh script.py --arg1 val1
# ./run.sh --dist-mode local script.py --arg1 val1
# ./run.sh --dist_mode=distributed script.py --arg1 val1

# parse arguments
DIST_MODE="local" # Set default mode
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --dist-mode|--dist_mode)
      DIST_MODE="$2"
      shift 2
      ;;
    --dist-mode=*|--dist_mode=*)
      DIST_MODE="${1#*=}"
      shift 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save non-flag arguments
      shift
      ;;
  esac
done

# restore positional arguments (script and its args)
set -- "${POSITIONAL_ARGS[@]}"

# inputs
INPUT_SCRIPT=$1
INPUT_ARGVS=${@:2}

echo "+ DIST_MODE: $DIST_MODE"
echo "+ INPUT_SCRIPT: $INPUT_SCRIPT"
echo "+ INPUT_ARGVS: $INPUT_ARGVS"

# if the input script is not found, exit
if [ ! -f "$INPUT_SCRIPT" ]; then
    echo "$INPUT_SCRIPT not found"
    exit 1
fi

# modules
nvidia-smi

# envs
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$PWD

# nums of GPUs
NGPUS=$(nvidia-smi -L | wc -l)
echo "+ NGPUS: $NGPUS"
if [ $NGPUS -eq 0 ]; then
    echo "No GPU found"
    exit 1
fi

# configure based on mode
if [ "$DIST_MODE" == "distributed" ]; then
    # slurm distributed setup
    NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    HEAD_NODE=${NODES_ARRAY[0]}
    HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')
    WORLD_SIZE=$SLURM_JOB_NUM_NODES
    SRUN_CMD="srun"

    echo "+ NODES_ARRAY: ${NODES_ARRAY[@]}"
    echo "+ HEAD_NODE: $HEAD_NODE"
else
    # local setup
    HEAD_NODE_IP="127.0.0.1"
    WORLD_SIZE=1
    SRUN_CMD=""

    echo "+ Running in LOCAL mode"
fi

echo "+ HEAD_NODE_IP: $HEAD_NODE_IP"
echo "+ WORLD_SIZE: $WORLD_SIZE"

# find an available port and use it
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close();')

# fire up
set -x
$SRUN_CMD torchrun \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv-endpoint $HEAD_NODE_IP:$PORT \
    --nnode $WORLD_SIZE \
    --nproc_per_node $NGPUS \
    $INPUT_SCRIPT \
    $INPUT_ARGVS
