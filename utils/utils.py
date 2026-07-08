import os
import re
import numpy as np


# Function for pulling Reynolds number ID
def get_re_load(re_targ,dir_name):
	files = os.listdir(dir_name)
	re_list = []
	for file in files:
		matches = re.findall(r'-?\d+', file)
		if matches:
			re_num = int(matches[0])
			re_list.append(re_num)
		else:
			pass
	ind = np.argmin(np.abs(np.array(re_list)-re_targ))
	return re_list[ind]

# Function to check for empty directories in training data
def clean_subdir_array(root_dir,subdir_array):
        subdir_list = []
        for subdir in subdir_array:
            if len(os.listdir(root_dir + str(subdir)))>1:
                subdir_list.append(subdir)
            else:
                pass
        return np.array(subdir_list)    
