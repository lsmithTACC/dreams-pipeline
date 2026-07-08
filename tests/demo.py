# Libraries
import functools
import os
import numpy as np
import torch
from models.FNO import FNO
from matplotlib import pyplot as plt

# Custom print
print = functools.partial(print, flush=True)

# Dataset Information: Time series specifications for each case
num_timesteps_prior = 10 
num_timesteps_forward = 40

# Intialize datasets
print('Loading data...')
root_dir='/work/10386/lsmith9003/vista/data/helmholtz_staircase/'
data_in = torch.tensor(np.load(root_dir+'data_in.npy'),dtype=torch.float32)
data_out = torch.tensor(np.load(root_dir+'data_out.npy'),dtype=torch.float32)
data_in_test = torch.tensor(np.load(root_dir+'data_in_test.npy'),dtype=torch.float32)
data_out_test = np.load(root_dir+'data_out_test.npy')

# Initialize FNO
print('Initializing FNO...')
fno_test = FNO(
	n_modes=(8,4),
	hidden_channels=32,
	n_epoch=50,
	batch_size=16,
	num_prior=num_timesteps_prior,
	num_forward=num_timesteps_forward,
	num_vector_components=data_in.shape[-1]
	)

# Train
print('Calling .train...')
fno_test.train(data_in,data_out)

# Save
print('Saving model...')
torch.save(fno_test,'fno.pth')
print('done')

# Eval
print('Evaluating...')
fno_test = torch.load('fno.pth',weights_only=False)
x0 = data_in_test
data_pred = fno_test.eval(x0)
print('Size of evaluated dataset:')
print(data_pred.shape)
print('Done.')

# Plot time series prediction
os.makedirs('images',exist_ok=True)
data_pred = data_pred[0]
data_pred = data_pred[...,1]
data_pred = data_pred.detach().numpy()
plot_max = np.max(data_pred)
data_out_test = data_out_test[...,1]
data_out_test = data_out_test[0]
for ind in range(0,len(data_pred)):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.contourf(data_pred[ind,:,:],np.linspace(-plot_max,plot_max,200),cmap='RdBu')
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted')
    plt.subplot(1,2,2)
    plt.contourf(data_out_test[ind,:,:],np.linspace(-plot_max,plot_max,200),cmap='RdBu')
    plt.xticks([])
    plt.yticks([])
    plt.title('True')
    plt.savefig("images/fig"+str(ind)+".png")
print('figures saved')
print(np.shape(data_pred))
print('done')

