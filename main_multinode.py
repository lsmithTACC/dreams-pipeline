# Libraries
import functools
import os
import numpy as np
import pandas as pd
import torch
import glob
import time
import json
import datetime
from matplotlib import pyplot as plt
from models.EPI_multinode import EPI 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.optim as optim


# -------------------- Setup Functions -------------------- #


def setup_distributed():
    """
    Initialize the distributed process group.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        rank, world_size, local_rank, master_addr, master_port
    """

    # Get local rank (GPU ID on the current node)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Get global rank and world size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Master address/port for multi-node training
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    # Initialize the process group
    # NCCL is faster for GPU, Gloo for CPU
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=1800)  # 30 minute timeout
    )
    
    # Set the current device to the local rank (specific GPU)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Optional: barrier to ensure all processes synchronize
    dist.barrier()
    
    return rank, world_size, local_rank, master_addr, master_port


def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_on_rank0(*args, **kwargs):
    """Print only on rank 0 process."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)        



# -------------------- Global Parameters -------------------- #


# Setup process group
rank, world_size, local_rank, master_addr, master_port = setup_distributed()

# Input parameters
state_name = 'New-Jersey' # options: District-of-Columbia, New-Jersey, North-Carolina, North-Dakota, Wisconsin
R0 = [1.0, 3.0, 5.0]
# We'll need a new way of parsing the initial infected json here, it's particular

# Simulation constants
num_features = 4
num_cases_per_dataset = 100
num_timesteps_total = 500
num_timesteps_prior = 1
num_timesteps_forward = num_timesteps_total - num_timesteps_prior

# Relevant file paths
metadata_master_path = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/metadata_master.csv'
county_pop_path = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/data/' + state_name + '/county_pop_by_age_' + state_name + '_2019-2023ACS.csv'
edge_matrix_path = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/data/' + state_name + '/' + state_name + '_Q4-2019_mobility-matrix.csv'
data_dir = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/SEIR-STOCH_Param_Sweep/' + state_name +'/'

# CUDA check
print_on_rank0('Checking Cuda...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print_on_rank0(torch.cuda.is_available())
print_on_rank0(f"Using device: {device}")


# -------------------- Data Loading -------------------- #

# Status update
print_on_rank0('Loading data...')

# Read metadata mater file
data = pd.read_csv(metadata_master_path)

# Simple for loop over rows
# Note that we need the 'sim_completion' flag to be checked here. Emily reran cases for which 100 simulations did not finish.
count = 0
num_nodes = 0
output_dirs = []

for ind in range(0,data.shape[0]):
	if data.loc[ind,'disease_R0'] in R0 and \
	data.loc[ind,'geo_region'] == state_name and \
	data.loc[ind,'sim_completion'] == 1:

		# Read initial infected json as a list of dictionaries
		# It should be read like: data_infected[i]['age_group']
		# where i is the seeded county, and the second key can be 'county', 'infected', or 'age_group'
		# Note that some cases seed multiple counties, so i will not always be 0.
		# For our initial subset, however, we're only looking for cases with a single seeded county, so len(data_infected)==1
		data_infected = json.loads(data.loc[ind,'initial_infected_json'])
		if len(data_infected) == 1.:
			output_dirs.append(data.loc[ind,'output_dir_path'])
			count += 1
			if not num_nodes:
				num_nodes = data.loc[ind,'geo_node_count']

# Initialize 
data_in = np.zeros((len(output_dirs)*num_cases_per_dataset,num_timesteps_prior,num_nodes,num_features))
data_out = np.zeros((len(output_dirs)*num_cases_per_dataset,num_timesteps_forward,num_nodes,num_features))

# Loop through identified files and load contents
case_count = 0
for output_dir in output_dirs:
	if case_count % 10 == 0 or case_count == len(output_dirs):
		print_on_rank0('Loading data for case: ' + str(case_count) + ' of ' + str(len(output_dirs)))
	case_dir = data_dir + output_dir
	files = glob.glob(os.path.join(case_dir,'node_*_batch-*.csv'))
	node_count = 0
	for file in files:
		data_sim = pd.read_csv(file).to_numpy()
		sim_id = data_sim[:,0]
		time_id = data_sim[:,1]
		data_sim = data_sim[:,5:(5+num_features)]
		for ind in range(0,num_cases_per_dataset):
			sim_inds = np.where(sim_id==ind)[0]
			data_ind = data_sim[sim_inds]
			if len(sim_inds) < num_timesteps_total:
				temp_tile = np.tile(data_ind[-1],(num_timesteps_total-len(sim_inds),1))
				data_ind = np.concatenate((data_ind,temp_tile))
			data_in[int(case_count*num_cases_per_dataset+ind),:,int(node_count),:] = data_ind[0:num_timesteps_prior]
			data_out[int(case_count*num_cases_per_dataset+ind),:,int(node_count),:] = data_ind[num_timesteps_prior:(num_timesteps_prior+num_timesteps_forward)]	
		node_count += 1
	case_count += 1

# Convert to tensor
data_in = torch.tensor(data_in,dtype=torch.float32)
data_out = torch.tensor(data_out,dtype=torch.float32)

# Move to GPU
#data_in = data_in.to(device)
#data_out = data_out.to(device)

# Create dataset, sampler, and data loader based on loaded epidemic data
dataset = TensorDataset(data_in, data_out)
sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# -------------------- Normalization -------------------- #

# Load county population matrix + sum over age group
county_pop = pd.read_csv(county_pop_path).to_numpy()
county_pop = np.sum(county_pop[:,1:],axis=-1)

# # Load edge matrix
edge_matrix = pd.read_csv(edge_matrix_path,skip_blank_lines=False,header=None).to_numpy()
# # We'll skip normalizing the edge weights for now, the max is close enough to 1 that we can revisit later.

# Convert edge matrix to format for GNN
edge_index = np.zeros((2,num_nodes*num_nodes))
edge_weight = np.zeros((num_nodes*num_nodes,))
count = 0
for ind_i in range(0,num_nodes):
	for ind_j in range(0,num_nodes):
		edge_index[:,count] = [ind_i,ind_j]
		edge_weight[count] = edge_matrix[ind_i,ind_j]
		count = count + 1
# Convert to tensor
edge_index = torch.tensor(edge_index,dtype=torch.int32)
edge_weight = torch.tensor(edge_weight,dtype=torch.float32)

# # Move to GPU
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)



# -------------------- Model Initialization -------------------- #

# Initialize model architecture
print_on_rank0('Initializing model...')
epi = EPI(
	edge_index=edge_index,
	edge_weight=edge_weight,
	county_pop=county_pop,
	hidden_channels=64,
	n_epoch= 1,
	batch_size=16,
	learning_rate = 1e-3,
	num_prior=num_timesteps_prior,
	num_total_timesteps=num_timesteps_forward,
	num_forward=1,
	num_features=num_features,
	num_nodes = num_nodes
	)

epi = epi.to(device)

# Wrap with DistributedDataParallel
epi = DDP(
        epi,
        device_ids=[local_rank],  # Only for single-node multi-GPU
        output_device=local_rank,  # Only for single-node multi-GPU
        find_unused_parameters=False,  # Set True if your model has unused params
    )

# -------------------- Model Training -------------------- #


# Load existing pre-trained model (Optional)
# epi = torch.load('fno_epidemic.pth',weights_only=False)
# epi.n_epoch = 30

# Train model
#epi.train(data_in,data_out)
optimizer = optim.Adam(list(epi.parameters()), lr = epi.module.learning_rate)

print_on_rank0('Starting training...')
for it in range(0, epi.module.n_epoch):

    loss_track = 0.0
    start_time = time.perf_counter()

    for batch_idx, (data_in, data_out) in enumerate(loader):
        data_in = data_in.to(device)
        data_out = data_out.to(device)
        optimizer.zero_grad()
        loss = epi(data_out,data_in)
        loss_track += loss.item()
        loss.backward()
        optimizer.step()

    if dist.is_initialized():
        loss_track_tensor = torch.tensor(loss_track).to(device)
        dist.all_reduce(loss_track_tensor, op=dist.ReduceOp.SUM)

    if dist.is_initialized():
        dist.barrier()
    
    end_time = time.perf_counter()
    print_on_rank0('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(loss_track) + '  |  ' + 'Time: ' + str(int(end_time-start_time)))

# Save final model
print_on_rank0('Saving final model...')
torch.save(epi,'epi.pth')
print_on_rank0('done')




# -------------------- Model Evaluation -------------------- #


# Status update
print_on_rank0('Evaluating...')

# Only perform evaluation/plotting on rank 0
if int(os.environ.get("RANK", 0)) == 0:

	# Load model weights
	epi = torch.load('epi.pth',weights_only=False)

	# Define input samples for evaluation (random)
	ind_sample = np.random.randint(low=0, high=len(data_in), size=100)
	sample = data_in[ind_sample]

	# Evaluate samples
	start_time = time.perf_counter()
	data_pred = epi.module.rollout(sample)
	end_time = time.perf_counter()
	print('Size of evaluated dataset:')
	print(data_pred.shape)
	print('Evaluation time:')
	print(str(end_time-start_time)+' seconds')
	print('Done.')

	# Collapse results into a global metric
	data_pred = data_pred.cpu().detach().numpy()
	data_out = data_out.cpu().detach().numpy()
	data_pred = np.sum(data_pred[:,:,:,0],axis=2)
	data_out = np.sum(data_out[:,:,:,0],axis=2)

	# Plot time series prediction (all simulations)
	plt.figure(figsize=(15,5))
	ybounds = [1e6,1e7]
	for sim_ind in range(0,len(data_pred)):
		plt.subplot(1,3,1)
		plt.plot(data_pred[sim_ind,:],alpha=0.5)
		plt.xlabel('Simulation Day')
		plt.ylabel('Susceptible')
		plt.title('Prediction')
		plt.ylim(ybounds)
		plt.subplot(1,3,2)
		plt.plot(data_out[np.random.randint(low=0,high=len(data_out)),:],alpha=0.5)
		plt.xlabel('Simulation Day')
		plt.ylabel('Susceptible')
		plt.title('Truth')
		plt.ylim(ybounds)
		if sim_ind == 0:
			plt.subplot(1,3,3)
			plt.plot(np.mean(data_out[ind_sample,:],axis=0),color=(0.3,0.3,0.3))
			plt.plot(np.mean(data_out[ind_sample,:],axis=0)+np.std(data_out[ind_sample,:],axis=0),linestyle=':',color=(0.3,0.3,0.3))
			plt.plot(np.mean(data_out[ind_sample,:],axis=0)-np.std(data_out[ind_sample,:],axis=0),linestyle=':',color=(0.3,0.3,0.3))
			plt.plot(np.mean(data_pred,axis=0),color=(0.6,0.74,0.92))
			plt.plot(np.mean(data_pred,axis=0)+np.std(data_pred,axis=0),linestyle=':',color=(0.6,0.74,0.92))
			plt.plot(np.mean(data_pred,axis=0)-np.std(data_pred,axis=0),linestyle=':',color=(0.6,0.74,0.92))
			plt.xlabel('Simulation Day')
			plt.ylabel('Susceptible')
			plt.title('Distribution Stats')
			plt.ylim(ybounds)
	plt.savefig("figures/susceptible.png")
	print('figures saved')
	print(np.shape(data_pred))
	print('done')


cleanup_distributed()
