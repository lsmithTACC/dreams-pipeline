# Libraries
import functools
import os
import numpy as np
import pandas as pd
import torch
import glob
import time
import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from models.DREAMS import DREAMS

# CUDA check
print('Checking Cuda...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(f"Using device: {device}")

# Dataset size/global constants
num_snapshots = 128
num_particles_gas = 1e5
num_particles_dm = 1e5
num_features_gas = 12
num_features_dm = 12

# Number of nearest neighbors in graph neural network
num_k = 10

# Define test/train split parameter (percentage of data devoted to testing)
test_size = 0.20


# -------------------- Data Loading -------------------- #
# 
# In this code block, you'll insert the code necessary to load your dataset into memory. The goal is to arrive at 
# an input and output dataset, each of shape:
# [batch, nodes, feat]
#
# In the case of the DREAMS data, batch corresponds to snapshot ID, nodes correspond to individual particles, and 
# features correspond to the gas/dark matter properties.
#
# In the space below, I generate a dummy data_in and data_out dataset of the appropriate size. These serve as a
# stand-in for your dataset so that the rest of the code template can proceed. However, this should be replaced
# before running the script.

# Status update
print('Loading data...')

# Initialize  
rng = np.random.default_rng()
data_in = rng.random((int(num_snapshots),int(num_particles_gas),int(num_features_gas)))
data_out = rng.random((int(num_snapshots),int(num_particles_dm),int(num_features_dm)))

# Convert to tensor
data_in = torch.tensor(data_in,dtype=torch.float32)
data_out = torch.tensor(data_out,dtype=torch.float32)

# Move to GPU
#data_in = data_in.to(device)
#data_out = data_out.to(device)



# -------------------- Normalization -------------------- #
#
# In the code block below, you will apply a normalization scheme to your data. The default normalization scheme is 
# to perform a Z-score over the batch and node dimensions. That is, we will compute an average/std for each feature
# across all nodes and samples, then normalize each feature based on this average/std.
#
# Alteratives to this approach involves changing the normalization type (e.g., swapping Z-score with MinMax scaling),
# or changing the dimension over which statistics are computing (e.g., computing an average/std only over the sample
# dimension).
#
# Also note that we store the computed mean/std for our output particles! This is necessary for reconstructing our 
# original data space later on.

# Status update
print('Normalizing data...')

# Loop over features for gas particles
for ind in range(num_features_gas):
    data_in[:,:,ind] = (data_in[:,:,ind] - torch.mean(data_in[:,:,ind]))/torch.std(data_in[:,:,ind])

# Loop over features for DM particles
mean_dm = torch.zeros(num_features_dm)
std_dm = torch.zeros(num_features_dm)
for ind in range(num_features_dm):
    mean_dm[ind] = torch.mean(data_out[:,:,ind])
    std_dm[ind] = torch.std(data_out[:,:,ind])
    data_out[:,:,ind] = (data_out[:,:,ind] - mean_dm[ind])/std_dm[ind]



# -------------------- Model Initialization -------------------- #
#
# In the code block below, you will initialize an instance of your ML model, the architecture of which is pulled
# from the model directory at the import stage. While the lines below shouldn't be changed much (except to tweak
# the batch size, model capacuty, or learning rate), the training loop, forward function, and loss function are 
# all defined in models/DREAMS.py. 

# Initialize model architecture
print('Initializing model...')
dreams = DREAMS(
	hidden_channels = 64,
	n_epoch = 5,
	batch_size = 16,
	learning_rate = 1e-3,
	num_nodes_gas = num_particles_gas,
	num_nodes_dm = num_particles_dm,
	num_features_gas = num_features_gas,
	num_features_dm = num_features_dm,
	num_k = num_k
	)

# Move model to GPU
dreams = dreams.to(device)



# -------------------- Model Training -------------------- #
#
# Below, we call a built-in train method for the DREAMS model. This function is defined in models/DREAMS.py and 
# outputs a train/test loss at each epoch as it trains the model weights. The following lines should remain
# unchanged, but be sure to check the train method in models/DREAMS.py to get an idea for the processes involved
# in each update.

# Set up test/train split
data_in_train, data_in_test, data_out_train, data_out_test = train_test_split(
	data_in, data_out, test_size=test_size, random_state=42
	)

# Train model
dreams.train(data_in_train,data_in_test,data_out_train,data_out_test)

# Save final model
print('Saving final model...')
torch.save(dreams,'dreams.pth')
print('done')




# -------------------- Model Evaluation -------------------- #
#
# In the code block below, you will evaluate the output of your trained model. This involves taking 10 random 
# from the input dataset, computing a prediction of the output for each, and comparing these outputs to the true
# data found in the original data_out tensor. The code below shows the syntax for sampling from the input tensor 
# and performing a forward prediction. You will need to select one of your feature and compare its predicted value
# with the true value using matplotlib (or any other plotting library). 
# 
# Note that this code block can be run as a way of evaluating existing models; that is, one can comment out the 
# training step above and run the evaluation by itself.

# Status update
print('Evaluating...')

# Load model weights
dreams = torch.load('dreams.pth',weights_only=False)

# Define input samples for evaluation (random)
ind_sample = np.random.randint(low=0, high=len(data_in), size=10)
sample = data_in[ind_sample]

# Evaluate samples
start_time = time.perf_counter()
data_pred = dreams(sample)
end_time = time.perf_counter()
print('Size of evaluated dataset:')
print(data_pred.shape)
print('Evaluation time:')
print(str(end_time-start_time)+' seconds')
print('Done.')

# Here you would add a plot that compares some aspect/feature of data_pred with data_out


