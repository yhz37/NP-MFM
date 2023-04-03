#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:50:42 2022

@author: y
"""
import numpy as np
from scipy.stats.qmc import LatinHypercube as splhs
from scipy.stats import qmc
from Gen_PBCM_CG import Gen_PBCM_CG
from Gen_CFD_CG import Gen_CFD_CG

def gen_data(case,number,eng):
    if '_9' in case:
        sampler = splhs(d=9)
        l_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        u_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif '_7' in case:
        sampler = splhs(d=7)
        l_bounds = [0, 0, 0, 0, 0, 0, 0]
        u_bounds = [1, 1, 1, 1, 1, 1, 1]
    samples_2D = sampler.random(n=number)
    
    samples_2D=qmc.scale(samples_2D, l_bounds, u_bounds)
    samples = samples_2D.reshape(number,1,-1)
    PBCM_samples = Gen_PBCM_CG(samples_2D,eng,case)
    PBCM_samples = PBCM_samples.reshape(number,1,-1)
    CFD_samples=[]
    for sample_i in samples:
        CFD_i = Gen_CFD_CG(sample_i,eng,case)
        CFD_samples = np.append(CFD_samples,CFD_i)
    
    CFD_samples = CFD_samples.reshape((number,1,-1))
    return samples,PBCM_samples,CFD_samples
    

# if '_9' in case:
#     Data = sio.loadmat('/home/y/y/Python/MFNN/Data/mCGG_9_Training_0.7_1_2000.mat')
#     TestData = sio.loadmat('/home/y/y/Python/MFNN/Data/mCGG_9_Testing_0.7_1_20.mat')
# elif '_7' in case:
#     Data = sio.loadmat('/home/y/y/Python/MFNN/Data/mCGG_7P_Training_0.7_1_500.mat')
#     TestData = sio.loadmat('/home/y/y/Python/MFNN/Data/mCGG_7P_Testing_0.7_1_20.mat')

# input = Data['input']
# PBCM_CG = Data['PBCM_CG']
# CFD_CG = Data['CFD_CG']
# PBCM_CG = np.transpose(PBCM_CG)
# CFD_CG = np.transpose(CFD_CG)

# numData = input.shape[0]
# input = input.reshape(numData,1,-1)
# PBCM_CG = PBCM_CG.reshape(numData, 1, -1)
# CFD_CG = CFD_CG.reshape(numData, 1, -1)


# Test_input = TestData['Test_input']
# Test_PBCM_CG = TestData['Test_PBCM_CG']
# Test_CFD_CG = TestData['Test_CFD_CG']
# Test_PBCM_CG = np.transpose(Test_PBCM_CG)
# Test_CFD_CG = np.transpose(Test_CFD_CG)
# Test_numData = Test_input.shape[0]
# Test_input = Test_input.reshape(Test_numData,1,-1)
# Test_PBCM_CG = Test_PBCM_CG.reshape(Test_numData, 1, -1)
# Test_CFD_CG = Test_CFD_CG.reshape(Test_numData, 1, -1)