#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:46:13 2022

@author: y
"""
import torch
from predictor_MF_Unet import predictor_MF_Unet
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
import numpy as np
def Cal_Aleatoric(val_set,case,MF_Unet):
    device = 'cpu'

    PBCM_CG_Val,CFD_CG_Val = val_set[:]
    PBCM_CG_Val = PBCM_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    if 'Aleatoric_He' in case:
        CFD_CG_Val_Predicted_mean,CFD_CG_Val_Predicted_std_a,CFD_CG_Val_Predicted_std_e,CFD_CG_Val_Predicted_std = predictor_MF_Unet(PBCM_CG_Val,MF_Unet,case,device) 
        CFD_CG_Val_Predicted_std_a = CFD_CG_Val_Predicted_std_a.detach().numpy()
        CFD_CG_Val_Predicted_std_e = CFD_CG_Val_Predicted_std_e.detach().numpy()
    else:
        CFD_CG_Val_Predicted_mean,CFD_CG_Val_Predicted_std = predictor_MF_Unet(PBCM_CG_Val,MF_Unet,case,device)
    PBCM_CG_Val = PBCM_CG_Val.detach().numpy()
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.detach().numpy()
    CFD_CG_Val_Predicted_mean = CFD_CG_Val_Predicted_mean.detach().numpy()
    CFD_CG_Val_Predicted_std = CFD_CG_Val_Predicted_std.detach().numpy()
    CFD_CG_Val = CFD_CG_Val.reshape(-1,100)
    CFD_CG_Val_Predicted_mean = CFD_CG_Val_Predicted_mean.reshape(-1,100)
    T = CFD_CG_Val_Predicted_mean.shape[0]
    Aleatoric = []
    for iii in range(100):
        def fitness_func(tau):    
            ll = logsumexp(-0.5 * tau * (CFD_CG_Val[:,iii] - CFD_CG_Val_Predicted_mean[:,iii])**2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)
            return -ll
        bnds =(0, 1e18)
        res = minimize_scalar(fitness_func, 50, method='Bounded', bounds=bnds)
        Aleatoric = np.append(Aleatoric,np.sqrt(1/res.x))
        
    Aleatoric = Aleatoric.reshape(1,100)
    return Aleatoric
