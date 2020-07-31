#!/bin/bash
python -m torch.distributed.launch --nnodes=$1 --node_rank=$2 --nproc_per_node=$3 --master_addr=$4 --master_port=$5 "${@:6}"