#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:55:19 2022

@author: y
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from predictor_MF_Unet import predictor_MF_Unet

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.15)

def MF_Unet_plot_Alea(val_set,case,MF_Unet,Aleatoric,device):
    criterion_test = nn.MSELoss()
    PBCM_CG_Val,CFD_CG_Val = val_set[:]
    PBCM_CG_Val = PBCM_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val_Predicted_mean,CFD_CG_Val_Predicted_std_e = predictor_MF_Unet(PBCM_CG_Val,MF_Unet,case,device)
    loss_Val = criterion_test(CFD_CG_Val_Predicted_mean, CFD_CG_Val)
    PBCM_CG_Val = PBCM_CG_Val.detach().numpy()
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.detach().numpy()
    CFD_CG_Val_Predicted_mean = CFD_CG_Val_Predicted_mean.detach().numpy()
    # print(np.sum(abs(CFD_CG_Val_Predicted_mean - CFD_CG_Val).reshape(-1,100), axis=1))

    CFD_CG_Val_Predicted_std_e = CFD_CG_Val_Predicted_std_e.detach().numpy()
    Aleatoric = Aleatoric.reshape(1,1,-1)
    CFD_CG_Val_Predicted_std = (CFD_CG_Val_Predicted_std_e**2+Aleatoric**2)**0.5

    xConc = np.linspace(0, 1, num=100)
    xConc = xConc.reshape([100,1])
    for i in range(10):
        fig = plt.figure()
        fig.set_size_inches(6, 5)

        plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)+2*Aleatoric.reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])+2*CFD_CG_Val_Predicted_std[i,:,:].reshape(-1), color = '#1f77b4',edgecolor = 'none', alpha = 0.3)
        plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)-2*Aleatoric.reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])-2*CFD_CG_Val_Predicted_std[i,:,:].reshape(-1), color = '#1f77b4',edgecolor = 'none', alpha = 0.3, label = 'Epistemic')
        plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)-2*Aleatoric.reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])+2*Aleatoric.reshape(-1), color = '#ff7f0e',edgecolor = 'none', alpha = 0.3, label = 'Aleatoric')        
        plt.plot(xConc,PBCM_CG_Val[i,:,0:100].reshape([100,1]), 'g--',linewidth=2,label = 'PBCM')
        plt.plot(xConc, CFD_CG_Val[i,:,:].reshape([100,1]), 'r-',linewidth=2,label = 'CFD')
        plt.plot(xConc,CFD_CG_Val_Predicted_mean[i,:,:].reshape([100,1]), 'b:',linewidth=2,label = 'NP-MFM')
        plt.ylabel("Concentration",fontsize=28,weight='bold')
        plt.xlabel("Normalized channel width",fontsize=28,weight='bold')
        fig.subplots_adjust(bottom=0.2)
        fig.subplots_adjust(left=0.2)
        plt.legend(shadow=True,prop={'weight':'bold','size':15})
        plt.xticks(fontsize=20,weight='bold')
        plt.yticks(fontsize=20,weight='bold')
        plt.xlim([0,1])
        plt.ylim([0,1])
        resolution_value = 1200
        plt.savefig("Test_{}_19.jpg".format(i+1), format="jpg", dpi=resolution_value)
        plt.show()
    # print(loss_Val.item())
    return loss_Val

