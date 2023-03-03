#!/bin/bash
NUM_PROC=$1
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"

