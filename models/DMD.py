# Import base python libraries
import math
import random
import os
import time

# Import the necessary array-handling libraries
import h5py
import torch
import numpy as np


# Dynamic Mode Decomposition 
class DMD(torch.nn.Module):

	# Initialization
	def __init__(
		self,
		n_modes: int,
		):

		super().__init__()
		self.n_modes = n_modes

	# Forward pass function
	def forward(self, x):
		x = np.transpose(x,[1,0])
		b = np.linalg.pinv(self.Phi)@x
		k = 1.
		temp = self.Phi @ np.diag(np.exp(k*np.log(np.diag(self.Lambda)))) @ b
		temp = np.real(temp).reshape(x.shape)
		return np.transpose(temp,[1,0])
	
	# Data packing function
	def data_packing(self,data):
		if data.shape[1] > 1:
			print('Error! DMD requires a sequence length of 1. Please reformat your dataset.')
			return 1
		else:
			pass

		if data.dim() > 3:
			collapsed_dim = torch.prod(torch.tensor(data.shape[2:])).item()
			data_packed = data.view(*data.shape[:2], collapsed_dim)
			self.original_data_shape = data.shape
		else:
			data_packed = data

		data_packed = torch.squeeze(data_packed)
		return data_packed.cpu().detach().numpy()

	# Data unpacking function
	def data_unpacking(self,data):
		if self.original_data_shape:
			data_unpacked = data.reshape(self.original_data_shape)
		else:
			data_unpacked = data
		return torch.tensor(data_unpacked,dtype=torch.float32)	


	# Training function
	def train(self, data_in, data_out):

		# Step 0: Pre-process the data matrix.
		X = self.data_packing(data_in)
		X = np.transpose(X,[1,0])
		Xp = self.data_packing(data_out)
		Xp = np.transpose(Xp,[1,0])

		# Step 1: Compute the SVD of the dataset and truncate
		r = self.n_modes
		U,Sigma,VT = np.linalg.svd(X,full_matrices=False)
		Ur = U[:,:r]
		Sigmar = np.diag(Sigma[:r])
		VTr = VT[:r,:]

		# Step 2: Compute the transformed linear operator
		Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xp @ VTr.T).T).T

		# Step 3: Eigen-decomposition of the transformed linear operator
		Lambda, W = np.linalg.eig(Atilde)
		Lambda = np.diag(Lambda)

		# Step 4: Compute DMD Modes
		Phi = Xp @ np.linalg.solve(Sigmar.T,VTr).T @ W
		alpha1 = Sigmar @ VTr[:,0]
		b = np.linalg.solve(W @ Lambda,alpha1)

		# Store the components of the DMD model
		self.Phi = Phi
		self.Lambda = Lambda

	# Evaluation Function
	def eval(self,x0):
		# x0 - the data point from which the prediction starts
		#    - assumed to be a single time step
		x_in = self.data_packing(x0)
		x_pred = self.forward(x_in)
		x_out = self.data_unpacking(x_pred)
		return x_out	

    
