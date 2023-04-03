#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:26:25 2022

@author: y
"""

import pickle
import matlab.engine
import os
import torch
import numpy as np


os.chdir('/home/ouj/projects/haizhou/To_Junlin/')

from Gen_PBCM_CG_NP import Gen_PBCM_CG_NP
from predictor_MF_Unet_NB import predictor_MF_Unet_NB


with open('/home/ouj/projects/haizhou/To_Junlin/MF_Unet2_9_MAP.pkl','rb') as f:  # Python 3: open(..., 'rb')
     MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device = pickle.load(f)

if 'eng' not in locals():
    os.chdir('/home/ouj/projects/haizhou/To_Junlin/Matlab_Code')
    eng = matlab.engine.start_matlab()
    os.chdir('/home/ouj/projects/haizhou/To_Junlin/')

# torch.cuda.set_device(3)

Pres_CG =  eng.DesignConc_9(matlab.double([0.7,0.8,0.9,1,0,0.8,0.2,0.6,0.4]),100)
Pres_CG = np.asarray(Pres_CG).transpose()
device = 'cpu'

def fitness_func(x):

    x = x.reshape(-1,9)
    PBCM_CG = Gen_PBCM_CG_NP(x,eng,case)
    x = x.reshape(-1,1,9)
    PBCM_CG = PBCM_CG.reshape(1,1,-1)
    All_input = np.concatenate((PBCM_CG, x), axis=2)
    # start_time_iter = time.time()

    CG_predict,_ = predictor_MF_Unet_NB(torch.from_numpy(All_input).to(device='cpu', dtype=torch.float32),MF_Unet,case,device)
    # print("--- %s seconds ---" % (time.time() - start_time_iter))

    CG_predict = CG_predict.detach().numpy()
    CG_predict = CG_predict.reshape(-1,100)
    jd = np.linalg.norm((CG_predict - Pres_CG).reshape(100,1), ord=1)

    return -jd.item()    

x = np.random.rand(9)
x = x.reshape(-1,9)
PBCM_CG = Gen_PBCM_CG_NP(x,eng,case)
x = x.reshape(-1,1,9)
PBCM_CG = PBCM_CG.reshape(1,1,-1)
All_input = np.concatenate((PBCM_CG, x), axis=2)
# start_time_iter = time.time()

CG_predict,_ = predictor_MF_Unet_NB(torch.from_numpy(All_input).to(device='cpu', dtype=torch.float32),MF_Unet,case,device)
# print("--- %s seconds ---" % (time.time() - start_time_iter))

y = fitness_func(x)