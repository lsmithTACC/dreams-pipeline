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

# Import PyTorch neural operator library
from neuralop.models import FNO as FNO_Base

# Import base model methods
from models.Base import Base_Model as Base_Model

# Import PyTorch Distributions Library
from torch.distributions import Normal, Gamma, Bernoulli, NegativeBinomial, Categorical, Poisson

# Import Pytorch-Geometric library
from torch_geometric.nn import GCNConv


# General GNN
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_nodes, activation):
        super(GCNModel, self).__init__()
        
        self.in_channels = in_channels
        self.activation = activation

        self.conv1 = GCNConv(in_channels,hidden_channels)
        #self.conv2 = GCNConv(hidden_channels,hidden_channels)
        self.conv3 = GCNConv(hidden_channels,hidden_channels)

        self.mu_head = nn.Sequential(
                        nn.Linear(in_channels,hidden_channels),
                        self.activation,
                        nn.Linear(hidden_channels,hidden_channels),
                        self.activation,
                        nn.Linear(hidden_channels,3)
                        ) 
        self.theta_head = nn.Sequential(
                        nn.Linear(in_channels,hidden_channels),
                        self.activation,
                        nn.Linear(hidden_channels,hidden_channels),
                        self.activation,
                        nn.Linear(hidden_channels,1)
                        ) 

        self.mu_node = nn.Sequential(
                        nn.Linear(hidden_channels,hidden_channels),
                        self.activation,
                        nn.Linear(hidden_channels,1)
                        )

        self.theta_node = nn.Sequential(
                        nn.Linear(hidden_channels,hidden_channels),
                        self.activation,
                        nn.Linear(hidden_channels,1)
                        )

    def forward(self, x, edge_index, edge_weight):

        # Parse outputs
        eps = 1e-3

        mu = F.softplus(self.mu_head(x)) + eps 
        theta = F.softplus(self.theta_head(x)) + eps

        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        x = self.conv3(x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        mu_S = F.softplus(self.mu_node(x)) + eps
        theta_S = F.softplus(self.theta_node(x)) + eps

        mu[...,0] = mu_S.squeeze(-1)
        theta[...,0] = theta_S.squeeze(-1)

        return mu, theta




# FNO: Fourier Neural Operator
class EPI(Base_Model):

    # Initialization
    def __init__(  
        self,
        edge_index,
        edge_weight,
        county_pop,
        hidden_channels: int,
        num_prior: int,
        num_forward: int,
        num_total_timesteps: int,
        num_features: int,
        num_nodes: int,
        n_epoch: int,
        batch_size: int,
        num_vector_components: int = 1,
        learning_rate: float = 1e-3
        ):

        super().__init__(n_epoch,batch_size,learning_rate)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.county_pop = county_pop
        self.hidden_channels = hidden_channels
        self.num_prior = num_prior
        self.num_forward = num_forward
        self.num_total_timesteps = num_total_timesteps
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_vector_components = num_vector_components
        self.activation = nn.ReLU()

        # Network for autoregressive map
        self.model = GCNModel(in_channels=self.num_features,
        	hidden_channels=self.hidden_channels,
        	num_nodes=self.num_nodes,
        	activation = self.activation)


    # Forward pass function
    def rollout(self, x0):

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        county_pop = torch.tensor(self.county_pop).to(device)

        # Loop through steps of trajectory
        trajectory = torch.zeros((x0.shape[0],self.num_total_timesteps,x0.shape[-2],x0.shape[-1]))
        trajectory = trajectory.to(device)
        x = x0
        for ind in range(0,self.num_total_timesteps,self.num_forward):

            # Forward prediction
            x_model = x/county_pop.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            mu, theta = self.model(x_model,self.edge_index,self.edge_weight)
            #mu = mu * county_pop.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            #theta = theta * county_pop.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            # Handle batch padding
            if ind+self.num_forward > self.num_total_timesteps:
                x = x[:,0:(self.num_total_timesteps-ind),:,:]
                mu = mu[:,0:(self.num_total_timesteps-ind),:,:]
                theta = theta[:,0:(self.num_total_timesteps-ind),:,:]

            # Update
            S_to_E = NegativeBinomial(logits=torch.log(mu[...,0])-torch.log(theta[...,0]),total_count=theta[...,0]).sample()
            E_to_I = Poisson(rate=mu[...,1]).sample()
            I_to_R = Poisson(rate=mu[...,2]).sample()

            dS = -S_to_E
            dE = S_to_E - E_to_I
            dI = E_to_I - I_to_R
            dR = I_to_R

            dx = torch.cat(( 
            	dS.unsqueeze(-1),
            	dE.unsqueeze(-1),
            	dI.unsqueeze(-1),
            	dR.unsqueeze(-1),
            	), dim=-1
            )

            x = x + dx

            # Store 
            trajectory[:,ind:(ind+self.num_forward),:,:] = x

        return trajectory


    # Loss function
    def forward(self, x_obs, x0):

        # Note: x_obs is the full observed trajectory, not the initial condition
        # x0 is the initial condition

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        county_pop = torch.tensor(self.county_pop).to(device)

        # Extract shapes from observed data x
        batch_size, T, node_num, output_dim = x_obs.shape
        log_likelihood = torch.tensor(0.0,device=device)

        # Loop through steps of the trajectory
        total_count = 0.
        x = x0
        for ind in range(0, self.num_total_timesteps, self.num_forward):

            # Forward prediction
            x_model = x/county_pop.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            mu, theta = self.model(x_model,self.edge_index,self.edge_weight)
            #mu = mu * county_pop.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            #theta = theta * county_pop.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            # Handle batch padding
            if ind+self.num_forward > self.num_total_timesteps:
                x = x[:,0:(self.num_total_timesteps-ind),:,:]
                mu = mu[:,0:(self.num_total_timesteps-ind),:,:]
                theta = theta[:,0:(self.num_total_timesteps-ind),:,:]

            # Compute delta from observations
            if ind+self.num_forward <= self.num_total_timesteps:
                obs = x_obs[:,ind:(ind+self.num_forward),:,:]
            else:
                obs = x_obs[:,ind:,:,:]
            delta = obs - x

            # Probabilities
            dSE_prob = NegativeBinomial(logits=torch.log(mu[...,0])-torch.log(theta[...,0]),total_count=theta[...,0]).log_prob(-delta[...,0])
            dEI_prob = Poisson(rate=mu[...,1]).log_prob(-delta[...,1]-delta[...,0])
            dIR_prob = Poisson(rate=mu[...,2]).log_prob(delta[...,-1])

            log_likelihood -= (dSE_prob.sum() + dEI_prob.sum() + dIR_prob.sum())/self.batch_size

            # Update x (teacher forcing)
            x = obs

        return log_likelihood







    # Data packing function
    def data_packing(self,data):
        if self.num_vector_components > 1:
            data_moved = torch.moveaxis(data,1,-1)
            data_flat = data_moved.reshape(*data_moved.shape[:-2],-1)
            data_packed = torch.moveaxis(data_flat,-1,1)
        else:
            data_packed = data
        return data_packed

    # Data un-packing function
    def data_unpacking(self,data):
        if self.num_vector_components > 1:
            data_moved = torch.moveaxis(data,1,-1)
            data_expanded = data_moved.reshape((*data_moved.shape[:-1], self.num_vector_components, self.num_forward))
            data_unpacked = torch.moveaxis(data_expanded,-1,1)
        else:
            data_unpacked = data
        return data_unpacked	

    # Training function 
    def train(self, data_in, data_out):

        print('Packing data...')
        data_in = self.data_packing(data_in)
        data_out = self.data_packing(data_out)
        sim_id = torch.arange(0,len(data_in))

        print('Setting up optimizer...')
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            ],lr = self.learning_rate)

        print('Starting training...')
        for it in range(0, self.n_epoch):

            ind_shuffle = torch.randperm(data_in.size()[0])
            data_in = data_in[ind_shuffle]
            data_out = data_out[ind_shuffle]
            sim_id = sim_id[ind_shuffle]
            loss_track = 0.0

            start_time = time.perf_counter()

            for ind in range(0,data_in.size()[0],self.batch_size):
                optimizer.zero_grad()
                ind_batch = range(ind,ind+self.batch_size)

                if ind+self.batch_size > data_in.size()[0]:
                    ind_batch = range(ind,data_in.size()[0])

                loss = self.loss(data_out[ind_batch], data_in[ind_batch])

                loss.backward()
                optimizer.step()
                loss_track += float(loss.item())
            end_time = time.perf_counter()
            print(
                'Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(loss_track) + '  |  ' + 'Time: ' + str(int(end_time-start_time))
            )

    # Evaluation function:
    def eval(self,x0,x):
        # x0 - the data point from which the prediction starts
        x_in = self.data_packing(x0)
        x = self.data_packing(x)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_pred = torch.zeros((len(x_in),self.num_forward,self.num_features))
        x_pred = x_pred.to(device)

        with torch.no_grad():
            for ind in range(0,len(x_in),self.batch_size):
                ind_batch = range(ind,ind+self.batch_size)
                if ind+self.batch_size > x_in.size()[0]:
                    ind_batch = range(ind,x_in.size()[0])	
                x_temp = self.forward(x_in[ind_batch])
                x_pred[ind_batch] = x_temp

        x_out = self.data_unpacking(x_pred)
        return x_out
