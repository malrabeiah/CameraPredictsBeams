'''
A data feeding class. It generates a list of data samples, each of which is
a tuple of a string (image path) and an integer (beam index), and it defines
a data-fetching method.
Author: Muhammad Alrabeiah
Aug. 2019
'''

import os
import random
from natsort import natsorted
import torch
import numpy as np
import random
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

################## Creating data samples ####################
def create_samples(root,shuffle,nat_sort = False):
    init_names = os.listdir(root) # List all sub-directories in root
    if nat_sort:
        sub_dir_names = natsorted(init_names) # sort directory names in natural order
                                              # (Only for directories with numbers for names)
    else:
        sub_dir_names = init_names

    class_to_ind = {name:int(name) for name in sub_dir_names}
    data_samples = []
    for sub_dir in sub_dir_names: # Loop over all sub-directories
        img_per_dir = os.listdir(root+'/'+sub_dir) # Get a list of image names from sub-dir # i
        if img_per_dir: # If img_per_dir is NOT empty
            for img_name in img_per_dir:
                sample = (root + '/' + sub_dir + '/' + img_name, class_to_ind[sub_dir])
                data_samples.append(sample)
    if shuffle:
        random.shuffle(data_samples)

    return data_samples
#############################################################

class DataFeed(Dataset):
    '''
    A class retrieving a tuple of (image,label). It can handle the case
    of empty classes (empty folders).
    '''
    def __init__(self,root_dir, nat_sort = False, transform=None, init_shuflle = True):
        self.root = root_dir
        self.samples = create_samples(self.root,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample[0])
        if self.transform:
            img = self.transform(img)
        label = sample[1]
        return (img,label)
