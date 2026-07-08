# Libraries
import functools
import os
import numpy as np
import torch
from scipy.io import loadmat
from models.DMD import DMD
from utils.utils import get_re_load, clean_subdir_array
from matplotlib import pyplot as plt

# Custom print
print = functools.partial(print, flush=True)

# Dataset Information: Reynolds number (FlowBench specific) and total number of cases
num_cases = 3
re_target = 200.

# Dataset Information: Time series specifications for each case
# Note: For LSTM, timesteps prior must equal timesteps forward
num_timesteps_prior = 1
num_timesteps_forward = num_timesteps_prior
num_timesteps_total = 242
num_total_batches = num_cases*(num_timesteps_total-num_timesteps_prior-num_timesteps_forward)

# Dataset Information: Root directory
root = '/scratch/10386/lsmith9003/data/FlowBench/FPO_NS_2D_1024x256/harmonics/'
subdir_array = np.arange(1,num_cases,1)
subdir_array = clean_subdir_array(root,subdir_array)
print('allocating data arrays')

# Intialize datasets
data_in = np.zeros((num_total_batches,num_timesteps_prior,256,1024,3))
data_out = np.zeros((num_total_batches,num_timesteps_forward,256,1024,3))
print('allocation done')
print('loading data')

# Load dataset into the standard format
count = 0
for subdir in subdir_array:
	dir_path = root + str(subdir)
	re_load = get_re_load(re_target,dir_path)

	print('Loading data for path: ' + dir_path + ' Re = ' + str(re_load))
	data_flow = np.load(root + str(subdir) + '/Re_' + str(re_load) + '.npz')['data']

	for ind in range(num_timesteps_prior,242-num_timesteps_forward):
		data_in[count,:,:,:,:] = np.squeeze(data_flow[ind-num_timesteps_prior:ind,:,:,:])
		data_out[count,:,:,:,:] = np.squeeze(data_flow[ind:ind+num_timesteps_forward,:,:,:])
		count += 1

# Convert to tensor
data_in = torch.tensor(data_in,dtype=torch.float32)
data_out = torch.tensor(data_out,dtype=torch.float32)

# Initialize LSTM
print('initializing DMD')
dmd_test = DMD(
		n_modes = 20,
	)
print('done')

# Train
print('Calling .train')
dmd_test.train(data_in,data_out)

# Save
print('Saving model...')
torch.save(dmd_test,'dmd_flowbench.pth')
print('done')

# Eval
print('Evaluating...')
dmd_test = torch.load('dmd_flowbench.pth',weights_only=False)
x0 = data_in[0:2]
data_pred = dmd_test.eval(x0)
print('Size of evaluated dataset:')
print(data_pred.shape)
print('Done.')

