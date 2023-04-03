#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:06:59 2022

@author: y
"""

import logging
import numpy as np
import os
import torch
import pickle
import matlab.engine

from train_MF_Unet import train_MF_Unet
from Gen_Infill_2 import Gen_Infill_2
from get_args_MF_Unet import get_args_MF_Unet
from FluidMixerDataset import FluidMixerDataset
from gen_data import gen_data
from MF_Unet_plot import MF_Unet_plot
from MF_Unet_plot_Alea import MF_Unet_plot_Alea
from Cal_Aleatoric import Cal_Aleatoric
        
        
if __name__ == '__main__':
    os.chdir('/home/y/y/Python/MFNN/Matlab_Code')
    eng = matlab.engine.start_matlab()
    os.chdir('/home/y/y/Python/MFNN/MF_Unet')

    torch.cuda.set_device(3)

    case = 'MF_Unet2_9_MAP'
    iteration = 19
    if '_9' in case:
        dim = 9
    elif '_7' in case:
        dim = 7
    
    # input,PBCM_CG,CFD_CG = gen_data(case,200,eng)
    # Test_input,Test_PBCM_CG,Test_CFD_CG = gen_data(case,50,eng)
    
    # with open('MF_Unet2_9_MAP_initial_200+50_0.1_1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([input,PBCM_CG,CFD_CG,Test_input,Test_PBCM_CG,Test_CFD_CG], f)
        
    # with open('/home/y/y/Python/MFNN/MF_Unet/Results/MF_Unet2_9_MAP_initial_200+50.pkl','rb') as f:  # Python 3: open(..., 'rb')
    #     input,PBCM_CG,CFD_CG,Test_input,Test_PBCM_CG,Test_CFD_CG = pickle.load(f)
    
    args = get_args_MF_Unet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # train_set = FluidMixerDataset(np.concatenate((PBCM_CG, input), axis=2), CFD_CG)
    # val_set = FluidMixerDataset(np.concatenate((Test_PBCM_CG,Test_input), axis=2), Test_CFD_CG)
    
    # MF_Unet,Final_lr,scheduler = train_MF_Unet(case,train_set,val_set,args.epochs,args.batch_size,args.lr,device,args.val,args.amp)
    # MF_Unets = []
    # MF_Unets.append(MF_Unet)
    with open('/home/y/y/Python/MFNN/MF_Unet/MF_Unet2_9_MAP_18.pkl','rb') as f:  # Python 3: open(..., 'rb')
          input,PBCM_CG,CFD_CG,MF_Unets,MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device = pickle.load(f)
    
    
    for infill_iter in range(18,iteration):
        infill,infill_PBCM,infill_CFD = Gen_Infill_2(case,MF_Unet,eng,args.Num_infill,device,input.reshape(-1,dim))
        input = np.append(input, infill, axis=0)
        PBCM_CG = np.append(PBCM_CG, infill_PBCM, axis=0)
        CFD_CG = np.append(CFD_CG, infill_CFD, axis=0)
        train_set = FluidMixerDataset(np.concatenate((PBCM_CG, input), axis=2), CFD_CG)
        MF_Unet,Final_lr,scheduler = train_MF_Unet(case,train_set,val_set,args.epochs,args.batch_size,args.lr,device,args.val,args.amp)
        MF_Unets.append(MF_Unet)
        torch.cuda.empty_cache()
        with open(case+'_{:d}.pkl'.format(infill_iter+1), 'wb') as f:  # Python 3: open(..., 'wb')
            if '_MAP' in case:
                pickle.dump([input,PBCM_CG,CFD_CG,MF_Unets,MF_Unet, Final_lr, scheduler,train_set,val_set,case,device], f)
    
    
    # loss_Train = MF_Unet_plot(train_set,case,MF_Unet,device) 
    # loss_Val = MF_Unet_plot(val_set,case,MF_Unet,device)
    Aleatoric = Cal_Aleatoric(val_set,case,MF_Unets[-1])
    loss_Train = MF_Unet_plot_Alea(train_set,case,MF_Unets[-1],Aleatoric,device)
    loss_Val = MF_Unet_plot_Alea(val_set,case,MF_Unets[-1],Aleatoric,device)
    with open(case+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        if '_MAP' in case:
            pickle.dump([input,PBCM_CG,CFD_CG,MF_Unets,MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device], f)
            
# import pickle
# from torch.utils.data import DataLoader
# from FluidMixerDataset import FluidMixerDataset
# from predictor_MF_Unet import predictor_MF_Unet
# from unet import UNet2_7, UNet2_9,UNet2_7_Aleatoric, UNet2_9_Aleatoric

    
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt


# with open('/home/y/y/Python/MFNN/MF_Unet/Results/MF_Unet2_7_MAP_Aleatoric.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     input,PBCM_CG,CFD_CG,MF_Unets,MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device = pickle.load(f)


