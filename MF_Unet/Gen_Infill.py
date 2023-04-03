#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:25:54 2022

@author: y
"""
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube as splhs
from Gen_PBCM_CG import Gen_PBCM_CG
from predictor_MF_Unet import predictor_MF_Unet
from Gen_CFD_CG import Gen_CFD_CG


def Gen_Infill(case,size,MF_Unet,eng,Num_infill,device):
    if '_9' in case:
        sampler = splhs(d=9)
    elif '_7' in case:
        sampler = splhs(d=7)
        
    samples_2D = sampler.random(n=2*size)
    samples = samples_2D.reshape(2*size,1,-1)
    PBCM_samples = Gen_PBCM_CG(samples_2D,eng,case)
    PBCM_samples = PBCM_samples.reshape(2*size,1,-1)
    All_input = np.concatenate((PBCM_samples, samples), axis=2)
    _,sigma = predictor_MF_Unet(torch.from_numpy(All_input).to(device='cpu', dtype=torch.float32),MF_Unet,case,device)
    sigma_mean = np.mean(np.mean(sigma.detach().numpy(),axis=1),axis=1)
    ind = np.argpartition(sigma_mean, -Num_infill)[-Num_infill:]
 
    infill = samples[ind]
    infill_PBCM=PBCM_samples[ind]
    infill_CFD=[]
    for infill_i in infill:
        infill_CFD_i = Gen_CFD_CG(infill_i,eng,case)
        infill_CFD = np.append(infill_CFD,infill_CFD_i)
    
    infill_CFD = infill_CFD.reshape((Num_infill,-1))
    return infill,infill_PBCM,infill_CFD
