# Import base python libraries
import math
import random
import os
import time
import datetime
import functools
print = functools.partial(print, flush=True)

# Import the necessary array-handling libraries
import h5py
import numpy as np

# Import Pytorch for neural network training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.parametrizations as param

# Import matplotlib for plotting 
import matplotlib.pyplot as plt

# Import base model methods
from models.Base import Base_Model as Base_Model

# Import loss functions
from loss_functions.MSE import MSE
from geomloss import SamplesLoss

# Import Pytorch-Geometric library
from torch_geometric.nn import GCNConv
from scipy.spatial import cKDTree


# General GNN
class GCNModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_nodes_in, num_nodes_out, activation):
        super(GCNModel, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.hidden_channels = hidden_channels
        self.activation = activation

        self.conv1 = GCNConv(in_channels,hidden_channels)
        self.conv2 = GCNConv(hidden_channels,hidden_channels)
        self.conv3 = GCNConv(hidden_channels,out_channels)


    def forward(self, x, edge_index):

        # Graph convolutions
        x = self.conv1(x, edge_index=edge_index)
        x = self.activation(x)

        x = self.conv2(x, edge_index=edge_index)
        x = self.activation(x)

        x = self.conv3(x, edge_index=edge_index)
        x = self.activation(x)

        return x



# FNO: Fourier Neural Operator
class DREAMS(Base_Model):

    # Initialization
    def __init__(  
        self,
        hidden_channels: int,
        num_nodes_gas: int,
        num_nodes_dm: int,
        num_features_gas: int,
        num_features_dm: int,
        num_k: int,    # number of nearest neighbors
        n_epoch: int,
        batch_size: int,
        learning_rate: float = 1e-3
        ):

        super().__init__(n_epoch,batch_size,learning_rate)
        self.hidden_channels = hidden_channels
        self.num_nodes_gas = num_nodes_gas
        self.num_nodes_dm = num_nodes_dm
        self.num_features_gas = num_features_gas
        self.num_features_dm = num_features_dm
        self.num_k = num_k
        self.activation = nn.ReLU()

        # Network for autoregressive map
        self.model = GCNModel(in_channels=self.num_features_gas,
        	out_channels=self.num_features_dm,
        	hidden_channels=self.hidden_channels,
        	num_nodes_in=self.num_nodes_gas,
        	num_nodes_out = self.num_nodes_dm,
        	activation = self.activation)

        self.loss = SamplesLoss("sinkhorn", p=2, blur=0.05, backend='multiscale')


    # Forward pass function
    def forward(self, x):

    	##### FINAL STEP: INSERT ESTIMATE OF EDGE_INDEX AND EDGE_WEIGHT HERE #####
    	# Seems like most particle simulators divide the domain into a volumetric grid, 
    	# and only search neighboring voxels
    	# cKDTree from scipy is also a really cool approach here

    	# Device
    	device = 'cuda' if torch.cuda.is_available() else 'cpu'

    	# Extract data shape
    	batch_size, N, _ = x.shape

    	# Initalize new objects for graph neural net
    	xs = [] # This will be the x tensor with flattened batch dimension
    	edges = [] # This will be the edge index
    	batch_vec = [] # This will keep track of which batch each entry belongs to (PyTorch GCN is not configured for a batch dimension)

    	# Loop through batches
    	offset = 0
    	for ind_batch in range(batch_size):
    	    pos_batch = x[ind_batch,:,0:3]
    	    tree = cKDTree(pos_batch)
    	    _, idx = tree.query(pos_batch, k=self.num_k+1) # idx has shape [N, k]
    	    idx = idx[:, 1:] # removes self from nearest neighbors arrays index

    	    src = np.repeat(np.arange(N),self.num_k) # an array of 1...N, repeated K times
    	    dst = idx.reshape(-1) # a flattened array of all neighbors

    	    src += offset # offsets the index to account for batch
    	    dst += offset

    	    edges.append(np.vstack([src, dst]))
    	    xs.append(x[ind_batch,:,:])
    	    batch_vec.append(np.full(N,ind_batch))

    	    offset += N

    	# Convert lists to tensors
    	x = np.vstack(xs)     
    	edge_index = np.hstack(edges)
    	batch = np.concatenate(batch_vec)
    	x = torch.tensor(x, dtype=torch.float32).to(device)
    	edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    	batch = torch.tensor(batch_vec, dtype=torch.long).to(device)

    	# Pass flattened data to and edge index to the GCN model
    	yPred = self.model(x, edge_index)

    	# Reshape output matrix so it can be used in loss function
    	yPred = yPred.reshape(int(batch_size), int(self.num_nodes_dm), int(self.num_features_dm))

    	return yPred

    # Training function 
    def train(self, data_in_train, data_in_test, data_out_train, data_out_test):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_out_train = data_out_train.to(device)
        data_out_test = data_out_test.to(device)

        print('Setting up optimizer...')
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            ],lr = self.learning_rate)

        print('Starting training...')
        loss_vec_train = []
        loss_vec_test = []
        for it in range(0, self.n_epoch):

            ind_shuffle = torch.randperm(data_in_train.size()[0])
            data_in = data_in_train[ind_shuffle]
            data_out = data_out_train[ind_shuffle]
            loss_track_train = 0.0
            loss_track_test = 0.0

            # Train loop 
            start_time = time.perf_counter()
            for ind in range(0,data_in.size()[0],self.batch_size):
                optimizer.zero_grad()
                ind_batch = range(ind,ind+self.batch_size)

                if ind+self.batch_size > data_in.size()[0]:
                    ind_batch = range(ind,data_in.size()[0])

                start_time = time.perf_counter()
                yPred = self.forward(data_in[ind_batch])
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                print(f"Forward pass time: {execution_time:.6f} seconds")

                start_time = time.perf_counter() 
                loss = self.loss(yPred, data_out[ind_batch])
                loss = loss.mean()
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                print(f"Loss calculation time: {execution_time:.6f} seconds")

                start_time = time.perf_counter()
                loss.backward()
                optimizer.step()
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                print(f"Backpropagation time: {execution_time:.6f} seconds")
                loss_track_train += float(loss.item())
                sss

            # Test loop
            for ind in range(0,data_in_test.size()[0], self.batch_size):
                ind_batch = range(ind,ind+self.batch_size)
                if ind+self.batch_size > data_in_test.size()[0]:
                    ind_batch = range(ind,data_in_test.size()[0])
                yPred_test = self.forward(data_in_test[ind_batch])    
                loss_track_test += float(self.loss(yPred_test, data_out_test[ind_batch]).mean().item()) 

            end_time = time.perf_counter()
            print(
                'Epoch: ' + str(it) + '  |  ' + 'Train Loss: ' + str(loss_track_train) + '  |  ' + 'Test Loss: ' + str(loss_track_test) + '  |  ' + 'Time: ' + str(int(end_time-start_time))
            )
            loss_vec_train.append(loss_track_train)
            loss_vec_test.append(loss_track_test)

        # Plot and save test/train loss behavior
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(range(0,self.n_epoch), loss_vec_train, label='train')
        ax.plot(range(0,self.n_epoch), loss_vec_test, label='test')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        ax.grid(True)
        fig.savefig('test_train_loss.png', dpi=300, bbox_inches="tight")
        plt.close(fig)




