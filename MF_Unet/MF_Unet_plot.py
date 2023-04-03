#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 22:16:50 2022

@author: y
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from predictor_MF_Unet import predictor_MF_Unet


def MF_Unet_plot(val_set,case,MF_Unet,device):
    criterion_test = nn.MSELoss()
    PBCM_CG_Val,CFD_CG_Val = val_set[:]
    PBCM_CG_Val = PBCM_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    if 'Aleatoric_He' in case:
        CFD_CG_Val_Predicted_mean,CFD_CG_Val_Predicted_std_a,CFD_CG_Val_Predicted_std_e,CFD_CG_Val_Predicted_std = predictor_MF_Unet(PBCM_CG_Val,MF_Unet,case,device) 
        CFD_CG_Val_Predicted_std_a = CFD_CG_Val_Predicted_std_a.detach().numpy()
        CFD_CG_Val_Predicted_std_e = CFD_CG_Val_Predicted_std_e.detach().numpy()
    else:
        CFD_CG_Val_Predicted_mean,CFD_CG_Val_Predicted_std = predictor_MF_Unet(PBCM_CG_Val,MF_Unet,case,device)
    loss_Val = criterion_test(CFD_CG_Val_Predicted_mean, CFD_CG_Val)
    PBCM_CG_Val = PBCM_CG_Val.detach().numpy()
    CFD_CG_Val = CFD_CG_Val.to(device='cpu', dtype=torch.float32)
    CFD_CG_Val = CFD_CG_Val.detach().numpy()
    CFD_CG_Val_Predicted_mean = CFD_CG_Val_Predicted_mean.detach().numpy()
    CFD_CG_Val_Predicted_std = CFD_CG_Val_Predicted_std.detach().numpy()



    xConc = np.linspace(0, 1, num=100)
    xConc = xConc.reshape([100,1])
    if 'Aleatoric_He' in case:

        for i in range(5):
            plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)+2*CFD_CG_Val_Predicted_std_a[i,:,:].reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])+2*CFD_CG_Val_Predicted_std[i,:,:].reshape(-1), color = '#1f77b4', alpha = 0.3, label = 'Epistemic')
            plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)-2*CFD_CG_Val_Predicted_std_a[i,:,:].reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])-2*CFD_CG_Val_Predicted_std[i,:,:].reshape(-1), color = '#1f77b4', alpha = 0.3, label = 'Epistemic')
            plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)-2*CFD_CG_Val_Predicted_std_a[i,:,:].reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])+2*CFD_CG_Val_Predicted_std_a[i,:,:].reshape(-1), color = '#ff7f0e', alpha = 0.3, label = 'Aleatoric')        
            plt.plot(xConc,PBCM_CG_Val[i,:,0:100].reshape([100,1]), 'g--', xConc, CFD_CG_Val[i,:,:].reshape([100,1]), 'r-',xConc,CFD_CG_Val_Predicted_mean[i,:,:].reshape([100,1]), 'b:')
            plt.ylabel("Concentration")
            plt.xlabel("Normalized channel width")
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.show()
    else:

        for i in range(5):

            
            plt.fill_between(xConc.reshape(-1), CFD_CG_Val_Predicted_mean[i,:,:].reshape(-1)+2*CFD_CG_Val_Predicted_std[i,:,:].reshape([-1]), CFD_CG_Val_Predicted_mean[i,:,:].reshape([-1])-2*CFD_CG_Val_Predicted_std[i,:,:].reshape(-1), color = '#1f77b4', alpha = 0.3, label = 'Epistemic')
            plt.plot(xConc,PBCM_CG_Val[i,:,0:100].reshape([100,1]), 'g--', xConc, CFD_CG_Val[i,:,:].reshape([100,1]), 'r-',xConc,CFD_CG_Val_Predicted_mean[i,:,:].reshape([100,1]), 'b:')
            plt.ylabel("Concentration",fontsize=16)
            plt.xlabel("Normalized channel width",fontsize=16)
            plt.legend(('2\u03C3','PBCM', 'CFD','Predicted'), shadow=True,fontsize=10)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.show()
            
    print(loss_Val.item())
    return loss_Val

