#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:16:30 2022

@author: y
"""

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube as splhs
from scipy.stats import qmc

from Gen_PBCM_CG import Gen_PBCM_CG
from predictor_MF_Unet import predictor_MF_Unet
from Gen_CFD_CG import Gen_CFD_CG

def get_smallest_dis(infill,exist_samples):
    distances = np.linalg.norm(infill-exist_samples,axis = 1)
    min_dis = np.amin(distances)
    return min_dis

def Gen_Infill_2(case,MF_Unet,eng,Num_infill,device,exist_samples):
    if '_9' in case:
        sampler = splhs(d=9)
        l_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        u_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif '_7' in case:
        sampler = splhs(d=7)
        l_bounds = [0, 0, 0, 0, 0, 0, 0]
        u_bounds = [1, 1, 1, 1, 1, 1, 1]
    size = 40000
    samples_2D = sampler.random(n=size)
    samples_2D = qmc.scale(samples_2D, l_bounds, u_bounds)
    samples = samples_2D.reshape(size,1,-1)
    PBCM_samples = Gen_PBCM_CG(samples_2D,eng,case)
    PBCM_samples = PBCM_samples.reshape(size,1,-1)
    All_input = np.concatenate((PBCM_samples, samples), axis=2)
    _,sigma = predictor_MF_Unet(torch.from_numpy(All_input).to(device='cpu', dtype=torch.float32),MF_Unet,case,device)
    sigma_mean = np.mean(np.mean(sigma.detach().numpy(),axis=1),axis=1)
    indices = np.argsort(sigma_mean)
    infill = samples[indices[-1]]
    infill_PBCM = PBCM_samples[indices[-1]]
    infill_min_dis = get_smallest_dis(infill,exist_samples)
    for iii in reversed(indices[:-1]):
        infill_candidate = samples[iii]
        infill_candidate_dis = np.linalg.norm(infill_candidate-infill,axis = 1)
        if np.all(infill_candidate_dis>infill_min_dis):
            infill = np.append(infill,infill_candidate, axis=0)
            infill_PBCM = np.append(infill_PBCM,PBCM_samples[iii], axis=0)
            infill_min_dis = np.append(infill_min_dis,get_smallest_dis(infill_candidate,exist_samples))
        if infill.shape[0]==Num_infill:
            break
    
    infill_CFD=[]
    for infill_i in infill:
        infill_CFD_i = Gen_CFD_CG(infill_i,eng,case)
        infill_CFD = np.append(infill_CFD,infill_CFD_i)
    infill = infill.reshape((Num_infill,1,-1))
    infill_PBCM = infill_PBCM.reshape((Num_infill,1,-1))
    infill_CFD = infill_CFD.reshape((Num_infill,1,-1))
    return infill,infill_PBCM,infill_CFD