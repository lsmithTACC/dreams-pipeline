#!/bin/bash

# Load CUDA module
module load cuda/12.8

# Load tacc-surrogates libraries
source /work/10386/lsmith9003/vista/python-envs/tacc-surrogates/bin/activate
export PYTHONPATH=/work/10386/lsmith9003/vista/python-envs/tacc-surrogates/lib/python3.11/site-packages/
export PATH=$PYTHONPATH:$PATH

# Run training/evaluation (single node)
python -W ignore main.py

# Run training/evaluation (multinode)
#HOST=$1
#NODES=$2
#LOCAL_RANK=${OMPI_COMM_WORLD_RANK}
#torchrun \
#	--nnodes=$NODES \
#	--nproc-per-node=1 \
#	--node_rank=${LOCAL_RANK} \
#	--master_addr=$HOST \
#	main_multinode.py

