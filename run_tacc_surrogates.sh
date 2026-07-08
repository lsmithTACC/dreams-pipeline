#!/bin/bash

#SBATCH -J run-dreams-ML-pipeline               # Job name
#SBATCH -o log.%j                         	 # Name of stdout output file (%j expands to jobId)
#SBATCH -p gh-dev                       # Queue name
#SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                                 # Total number of mpi tasks requested
#SBATCH -t 2:00:00                          # Run time (hh:mm:ss)
#SBATCH -A XXXXXXXX							 # Project charge code

# Load CUDA module
module load cuda/12.8

# Load tacc-surrogates libraries
source /work/10386/lsmith9003/vista/python-envs/tacc-surrogates/bin/activate
export PYTHONPATH=/work/10386/lsmith9003/vista/python-envs/tacc-surrogates/lib/python3.11/site-packages/
export PATH=$PYTHONPATH:$PATH

# Run training/evaluation (single node)
python -W ignore main.py

# Run training/evaluation (multinode)
#MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#mpirun -np $SLURM_NTASKS --map-by ppr:1:node run_tacc_surrogates_idev.sh $MASTER_ADDR $SLURM_JOB_NUM_NODES
