import numpy as np
import torch
import mat73
import scipy.io

from utils import *

# This function is adapted from https://github.com/SNU-LIST/QSMnet/blob/master/Code/training_data_patch.py
## Data path (Train, .mat file) 
FILE_PATH_INPUT1 = ''
FILE_PATH_MASK1 = ''
FILE_PATH_MASKmimi1 = ''
FILE_PATH_LABEL1 = ''


## Patch Constant Variables
PS = 64  # Patch Size
patch_num = [5, 4, 4] 
sub_num = 1
dir_num = 1

# Initialize the lists for storing patches
patches = []
patches_mask = []
patches_maskmimi = []
patches_label = []


# Patch Generation

def load_mat_file(file_path, key):
    data = mat73.loadmat(file_path)
    matrix = data[key]
    if matrix.ndim == 3:
        matrix = np.expand_dims(matrix, axis=3)
    return matrix

def append_patches(patches_list, matrix, strides, PS, idx):
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                patches_list.append(matrix[
                                    (i*strides[0]):(i*strides[0]+PS),
                                    (j*strides[1]):(j*strides[1]+PS),
                                    (k*strides[2]):(k*strides[2]+PS),
                                    idx])
    return patches_list

for dataset_num in range(1, sub_num+1):
    input_mat = load_mat_file(FILE_PATH_INPUT1, 'Lim_up')
    mask_mat = load_mat_file(FILE_PATH_MASK1, 'Mask_lim')
    mask_mimi_mat = load_mat_file(FILE_PATH_MASKmimi1, 'Mask_mimi')
    label_mat = load_mat_file(FILE_PATH_LABEL1, 'Full')
    
    matrix_size = input_mat.shape
    
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
    
    matrices_dict = {
        '': input_mat,
        '_mask': mask_mat,
        '_maskmimi': mask_mimi_mat,
        '_label': label_mat
    }

    for idx in range(dir_num):
        for key_suffix, matrix in matrices_dict.items():
            locals()[f'patches{key_suffix}'] = append_patches(locals()[f'patches{key_suffix}'], matrix, strides, PS, idx)


# Save to the train .mat file
scipy.io.savemat('file_train.mat', {
    'patches': patches,
    'patches_mask': patches_mask,
    'patches_maskmimi': patches_maskmimi,
    'patches_label': patches_label
})
