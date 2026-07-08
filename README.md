# TACC-Surrogates

TACC-Surrogates is a series of python wrappers and environment configuration files intended to streamline the training and evaluation of surrogate models on TACC systems. In this repository, we collect a number of popular surrogate model architectures, and outfit each with a common initialization, training, and evaluation method.

The main advantage of TACC-Surrogates is its common data structure. Our intent is for the user
to freely swap model architectures without having to reformat their dataset. Toward this end, we adopt the following standard shape for input/output datasets:
```
(batch, sequence_length, features)
```
Here, batch corresponds to the batch dimension, sequence_length corresponds to the number
of previous timesteps (for input data) or the number of forward time steps (for output data),
and features corresponds to the data features associated with each timestep. 
These features can be organizaed as a single, flattened array, or as 
a multi-dimensional grid (if one chooses an architecture that exploits spatial correlations). 

Each member of the ```models``` subdirectory is configured to accept datasets in the standard form. 


## Installing on Vista

Installing tacc-surrogates on TACC's Vista system is as simple as cloning this repository:
```
git clone https://github.com/lsmithTACC/tacc-surrogates.git
```

And executing a requirements install with pip:
```
cd tacc-surrogates
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
The ```tests/demo.py``` script serves as a quick way to confirm that your environment is configured correctly. This script loads 
an [acoustic scattering dataset](https://polymathic-ai.org/the_well/datasets/helmholtz_staircase/) and fits the data to a Fourier Neural Operator (FNO) model. Training should complete in a few minutes at most:
```
python -W ignore tests/demo.py 
```

If you encounter any issues with the requirements file, or if you want to offload environment maintenance, we are also hosting a pre-installed version of the environment on a TACC staff account. You can activate it with the following commands (permissions are set such that any user can activate):
```
module load cuda/13.1
source /work/10386/lsmith9003/python-envs/tacc-surrogates/bin/activate
```

## Training on Existing Architectures

The user can train on any architecture in the ```models``` sub-directory with just a few function calls. Namely, if the user has already loaded the correct envionrment and shaped their data according to the standard format, they need only (1) initialize an instance of the model architecture, and (2) call the built-in .train method. 

As an example, if one wishes to train their dataset using the Fourier Neural Operator (FNO), 
they need only run the following commands:

```
# 1) Initialize FNO
fno_test = FNO(
        n_modes=(16,16),
        hidden_channels=32,
        n_epoch=2,
        batch_size=16,
        num_prior=num_timesteps_prior,
        num_forward=num_timesteps_forward,
        num_vector_components=data_in.shape[-1]
        )

# 2) Train
fno_test.train(data_in,data_out)
```

The arguments for initialization (n_epoch, batch_size, etc.) can be found by looking at the relevant file within the ```models``` sub-directory. Most initialization steps are quite similar.

If there is any confusion regarding the initialization/training of a given architecture, the ```tests``` sub-directory contains a complete submission script for certain models, with the [FlowBench dataset](https://baskargroup.bitbucket.io/) serving as a test bed. This sub-directory will be updated as tests are completed on new architectures.


## List of Architectures to be Added

The following is a master list of all architectures slated to be added to the ```models``` sub-directory. An (x) will be placed next to architectures that have been successfully uploaded and tested.

- DMD (x)
- Neural ODE (x) 
- FNO (x)
- LSTM (x)
- GNN (x)
- Transformer (in progress)
- U-Net
- Deep-O-Net
