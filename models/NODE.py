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

# Import PyTorch differential equation library
from torchdiffeq import odeint

# Import base model methods
from models.Base import Base_Model as Base_Model

# NODE: Neural Ordinary Differential Equation
class NODE(Base_Model):

	# Initialization:
	def __init__(
		self,
		num_prior: int,			# must be = 1 for neural ODE
		num_forward: int,
		num_vector_components: int,
		grid_dimension: int, 		# options are 1 and 2
		n_epoch: int,
		batch_size: int,
		kernel_size: int = 5,
		padding: str = 'same',
		stride: int = 1,
		activation: str = 'relu', 	# options are 'relu' and 'tanh'
		learning_rate: float = 1e-3
		):

			super().__init__(n_epoch,batch_size,learning_rate)

			self.num_prior = num_prior
			self.num_forward = num_forward
			self.num_vector_components = num_prior*num_vector_components
			self.n_channel = num_vector_components
			self.kernel_size = kernel_size
			self.padding = padding
			self.stride = stride
			if activation == 'relu':
				self.activation = nn.ReLU()
			elif activation == 'tanh':
				self.activation = nn.Tanh()

			# Neural network definition
			if grid_dimension == 1:
				self.ode = nn.Sequential(
					nn.Conv1d(in_channels=self.n_channel,out_channels=2*self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding),
					self.activation,
					nn.Conv1d(in_channels=2*self.n_channel,out_channels=4*self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding),
					self.activation,
					nn.Conv1d(in_channels=4*self.n_channel,out_channels=2*self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding),
					self.activation,
					nn.Conv1d(in_channels=2*self.n_channel,out_channels=self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)
					)
			elif grid_dimension == 2:	
				self.ode = nn.Sequential(
					nn.Conv2d(in_channels=self.n_channel,out_channels=2*self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding),
					self.activation,
					nn.Conv2d(in_channels=2*self.n_channel,out_channels=4*self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding),
					self.activation,
					nn.Conv2d(in_channels=4*self.n_channel,out_channels=2*self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding),
					self.activation,
					nn.Conv2d(in_channels=2*self.n_channel,out_channels=self.n_channel,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)
					)

			self.loss_function = nn.MSELoss()

	# Forward pass function:
	def forward(self, t, x):
		return self.ode(x)	

	# Data packing function
	def data_packing(self,data):
		if self.num_vector_components > 1:
			data_moved = torch.moveaxis(data,1,-1)
			data_flat = data_moved.reshape(*data_moved.shape[:-2],-1)
			data_packed = torch.moveaxis(data_flat,-1,1)
		else:
			data_packed = data
		return data_packed	

	def data_unpacking(self,data):
		if self.num_vector_components > 1:
			data_moved = torch.moveaxis(data,1,-1)
			data_expanded = data_moved.reshape((*data_moved.shape[:-1], self.num_vector_components, self.num_forward))
			data_unpacked = torch.moveaxis(data_expanded,-1,1)
		else:
			data_unpacked = data
		return data_unpacked			

	# Training function:
	def train(self, data_in, data_out):

		optimizer = optim.Adam(list(self.ode.parameters()), lr = self.learning_rate)

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

				y_pred = odeint(self,data_in[ind_batch],torch.arange(0,self.num_forward+1,dtype=torch.float32))
				y_pred = y_pred[1:] 
				y_pred = y_pred.swapdims(1,2)
				y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],*y_pred.shape[2:]) 
				y_pred = y_pred.swapdims(0,1)
				loss = self.loss_function(y_pred,data_out[ind_batch])
				loss.backward()
				optimizer.step()
			
			print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))

	# Evaluation function:
	def eval(self,x0):
		# x0 - the data point from which the prediction starts
		x_in = self.data_packing(x0)
		x_pred = odeint(self,x_in,torch.arange(0,self.num_forward+1,dtype=torch.float32))
		x_pred = x_pred[1:] 
		x_pred = x_pred.swapdims(1,2)
		x_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1],*x_pred.shape[2:])
		x_pred = x_pred.swapdims(0,1)
		x_out = self.data_unpacking(x_pred)
		return x_out



