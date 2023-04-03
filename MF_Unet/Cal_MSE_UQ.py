# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:39:31 2022

@author: haizhouy
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from predictor_MF_Unet import predictor_MF_Unet

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)

def Cal_MSE_UQ(val_set,case,MF_Unet,Aleatoric,device):
    PBCM_CG_Val,CFD_CG_Val = val_set[:]
    PBCM_CG_Val = PBCM_CG_Val.to(device=device, dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.to(device=device, dtype=torch.float32)
    CFD_CG_Val_Predicted_mean,CFD_CG_Val_Predicted_std_e = predictor_MF_Unet(PBCM_CG_Val,MF_Unet,case,device)
    diff2 = (CFD_CG_Val_Predicted_mean- CFD_CG_Val)**2
    MSE = torch.mean(diff2,[1,2])
    Aleatoric = Aleatoric.reshape(1,1,-1)
    total_UQ = torch.mean((CFD_CG_Val_Predicted_std_e**2+Aleatoric**2),[1,2])
    Epstemic = torch.mean(CFD_CG_Val_Predicted_std_e**2,[1,2])
    Aleatoric = np.mean(Aleatoric**2,(1,2))

    Epstemic = Epstemic.detach().numpy()
    MSE=MSE.detach().numpy()
    total_UQ=total_UQ.detach().numpy()
    return MSE,Aleatoric,Epstemic,total_UQ
