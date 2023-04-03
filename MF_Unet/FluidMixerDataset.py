#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:38:49 2022

@author: y
"""
from torch.utils.data import Dataset
import torch


class FluidMixerDataset(Dataset):    

    # Initialize your data
    def __init__(self, x, y):        
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len     