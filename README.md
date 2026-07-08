## TACC-Surrogates: DREAMS pipeline

TACC-Surrogates is a series of python wrappers and environment configuration files intended to streamline the training and evaluation of surrogate models on TACC systems. This repository is a fork of the original tacc-surrogates library. It has been configured to train neural networks based on datasets derived from the DREAMS project (Dark Matter and Astrophysics with Machine learning and Simulations).

The following provides step-by-step instructions for running the pipeline on Vista:

1. Log in to Vista and navigate to your scratch directory:
```
ssh <username>@vista.tacc.utexas.edu
cd $SCRATCH
```

2. Create a copy of the source code:
```
git clone https://github.com/lsmithTACC/dreams-pipeline.git
```

3. Add your data loading/evaluation code snippets to `main.py`. Detailed instructions for this step can be found within the `main.py` script itself.

4. Add your allocation ID to the job submission script `run_tacc_surrogates.sh`.

5. Submit the job via sbatch:
```
sbatch run_tacc_surrogates.sh
```
This submission script will write the code's progress to a slurm log file. You can monitor your training progress by passing the log file to `tail`, or by simply opening the log file with `vim`. All errors will be traced within this log file. Once training has completed, the script should generate a png file (test_train_loss.png) that plots learning curves for the train and test data, and you should see a model file (dreams.pth) appear within the dreams-pipeline directory.

6. Optional: as an alterative to sbatch, you may want to run the pipeline in an interactive session. This is useful for scenarios in which you need to run the pipeline several times over a short period, such as debugging. Toward this end we have included a version of the job submission script intended for use in an interaction session:
```
idev -n 1 -N 1 -p gh-dev -t 02:00:00
./run_tacc_surrogates_idev.sh
```

Note that we have already set up an environment with all the dependencies needed to run the pipeline. We've set permissions such that any TACC user can access the environment via:
```
source /work/10386/lsmith9003/vista/python-envs/tacc-surrogates/bin/activate
```
Both job submission scripts (run_tacc_surrogates.sh and run_tacc_surrogates_idev.sh) are set to activate this environment by default.
