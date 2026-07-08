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

# Import PyTorch neural operator library
from neuralop.models import FNO as FNO_Base

# Import PyTorch normalizing flows library
import normflows as nf
from normflows.flows.affine.coupling import Flow
from normflows.flows.reshape import Split, Merge

# Import base model methods
from models.Base import Base_Model as Base_Model



# FNO: Fourier Neural Operator
class FNO(Base_Model):

	# Initialization
	def __init__(
		self,
		n_modes: tuple,
		hidden_channels: int,
		num_prior: int,
		num_forward: int,
		num_features: int,
		num_vector_components: int,
		n_epoch: int,
		batch_size: int,
		learning_rate: float = 1e-3
		):

		super().__init__(n_epoch,batch_size,learning_rate)
		self.n_modes = n_modes
		self.hidden_channels = hidden_channels
		self.num_prior = num_prior
		self.num_forward = num_forward
		self.num_features = num_features
		self.num_vector_components = num_vector_components

		self.activation = nn.ReLU()
		#self.activation = nn.Tanh()

		self.model = FNO_Base(
			n_modes = self.n_modes,
			hidden_channels = self.hidden_channels,
			in_channels = self.num_prior*self.num_vector_components,#int(self.num_forward/2)*self.num_vector_components,
			out_channels = self.num_forward*self.num_vector_components)

		# Loss function
		self.loss_function = nn.MSELoss()



	# Forward pass function
	def forward(self, x, sim_id):

		x = self.model(x)

		return x 


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

			for ind in range(0,data_in.size()[0],self.batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+self.batch_size)
				
				if ind+self.batch_size > data_in.size()[0]:
					ind_batch = range(ind,data_in.size()[0])

				y_pred = self.forward(data_in[ind_batch],sim_id[ind_batch])

				loss = self.loss_function(y_pred,data_out[ind_batch])

				loss.backward()
				optimizer.step()
			print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))

	# Evaluation function:
	def eval(self,x0):
		# x0 - the data point from which the prediction starts
		sim_id = torch.arange(0,len(x0))
		x_in = self.data_packing(x0)

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		x_pred = torch.zeros((len(x_in),self.num_forward,self.num_features))
		x_pred = x_pred.to(device)

		with torch.no_grad():
			for ind in range(0,len(x_in),self.batch_size):

				ind_batch = range(ind,ind+self.batch_size)

				if ind+self.batch_size > x_in.size()[0]:
					ind_batch = range(ind,x_in.size()[0])	

				x_temp = self.forward(x_in[ind_batch],sim_id[ind_batch])
				x_pred[ind_batch] = x_temp
			
		x_out = self.data_unpacking(x_pred)
		return x_out			
			
	
