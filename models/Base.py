# Import base python libraries
import math
import random
import os
import time

# Import the necessary array-handling libraries
import h5py
import numpy as np

# Import Pytorch for neural network training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import default loss function
from loss_functions.MSE import MSE

import torch.distributed as dist


def print_on_rank0(*args, **kwargs):
    """Print only on rank 0 process."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)      


# Base Model: template class that includes methods common to all architectures
class Base_Model(torch.nn.Module):

	# Initialization
	def __init__(
		self,
		n_epoch: int,
		batch_size: int,
		learning_rate: float = 1e-3
		):

		super().__init__()
		self.n_epoch = n_epoch
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.model = nn.Identity()
		self.loss_function = MSE()


	# Forward pass function
	def forward(self, x):
		return self.model(x)
			

	# Data packing function
	def data_packing(self,data):
		return data	

	def data_unpacking(self,data):
		return data	

	# Training function 
	def train(self, data_in, data_out):

		optimizer = optim.Adam(list(self.model.parameters()), lr = self.learning_rate)

		print('Packing data...')
		data_in = self.data_packing(data_in)
		data_out = self.data_packing(data_out)

		print('Starting training...')
		for it in range(0, self.n_epoch):

			ind_shuffle = torch.randperm(data_in.size()[0])
			data_in = data_in[ind_shuffle]
			data_out = data_out[ind_shuffle]

			for ind in range(0,data_in.size()[0],self.batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+self.batch_size)
				
				if ind+self.batch_size > data_in.size()[0]:
					ind_batch = range(ind,data_in.size()[0])

				y_pred = self.forward(data_in[ind_batch])
				loss = self.loss_function(y_pred,data_out[ind_batch])
				loss.backward()
				optimizer.step()

			print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))


	# Training function (multinode)
	def train_multinode(self, data_loader):

		optimizer = optim.Adam(list(self.model.parameters()), lr = self.learning_rate)

		print_on_rank0('Starting training...')
		for it in range(0, self.n_epoch):

			loss_track = 0.0

			for batch_idx, (data_in, data_out) in enumerate(data_loader):
				optimizer.zero_grad()
				y_pred = self.forward(data_in)
				loss = self.loss_function(y_pred,data_out)
				loss_track += loss.item()
				loss.backward()
				optimizer.step()

			if dist.is_initialized():
				loss_track_tensor = torch.tensor(loss_track).to(device)
				dist.all_reduce(loss_track_tensor, op=dist.ReduceOp.SUM)

			if dist.is_initialized():
				dist.barrier()	

			print_on_rank0('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss_track.item())))
			

	# Evaluation function:
	def eval(self,x0):
		# x0 - the data point from which the prediction starts
		x_in = self.data_packing(x0)
		x_pred = self.forward(x_in)
		x_out = self.data_unpacking(x_pred)
		return x_out			
