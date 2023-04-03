#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:45:18 2022

@author: y
"""
import torch
def log_gaussian_loss(output, target, s):
    loss = torch.mean(0.5*torch.exp(-s)*(output-target)**2+0.5*s)
    return loss