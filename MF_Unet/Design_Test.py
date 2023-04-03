#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:44:57 2022

@author: y
"""

import pygad
import pickle
import matlab.engine
import os
import torch
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import time
import matplotlib.pyplot as plt


os.chdir('/home/y/y/Python/MFNN/MF_Unet')

from Gen_PBCM_CG_NP import Gen_PBCM_CG_NP
from predictor_MF_Unet_NB import predictor_MF_Unet_NB
from Gen_CFD_CG import Gen_CFD_CG
# from torch.utils.data import DataLoader
# from FluidMixerDataset import FluidMixerDataset
# from unet import UNet2_7, UNet2_9,UNet2_7_Aleatoric, UNet2_9_Aleatoric
# from Gen_Infill import Gen_Infill
# from Gen_Infill_2 import Gen_Infill_2
# from FluidMixerDataset import FluidMixerDataset
# from train_MF_Unet import train_MF_Unet
# from get_args_MF_Unet import get_args_MF_Unet


# os.chdir('/home/y/y/Python/MFNN/Matlab_Code')
# eng = matlab.engine.start_matlab()
# os.chdir('/home/y/y/Python/MFNN/MF_Unet')

with open('E:\OneDrive - University of South Carolina\Python\MFNN\MF_Unet\Results\MF_Unet2_9_MAP.pkl','rb') as f:  # Python 3: open(..., 'rb')
      MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device = pickle.load(f)
# with open('/home/y/y/Python/MFNN/MF_Unet/MF_Unet2_9_MAP.pkl','rb') as f:  # Python 3: open(..., 'rb')
#       input,PBCM_CG_Train,CFD_CG_Train,MF_Unets,MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device = pickle.load(f)
if 'eng' not in locals():
    os.chdir('E:\OneDrive - University of South Carolina\Python\MFNN\Matlab_Code')
    eng = matlab.engine.start_matlab()
    os.chdir('E:\OneDrive - University of South Carolina\Python\MFNN\MF_Unet')

# torch.cuda.set_device(3)

Pres_CG =  eng.DesignConc_9(matlab.double([0.7,0.8,0.9,1,0,0.8,0.2,0.6,0.4]),100)
Pres_CG = np.asarray(Pres_CG).transpose()
device = 'cpu'

def fitness_func(x,index):

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

num_generations = 1000
num_parents_mating = 100
sol_per_pop = 200
num_genes = 9
gene_space = [{'low': 0.7, 'high': 1},{'low': 0.7, 'high': 1},{'low': 0.7, 'high': 1},{'low': 0, 'high': 1},{'low': 0, 'high': 1},{'low': 0, 'high': 1},{'low': 0, 'high': 1},{'low': 0, 'high': 1},{'low': 0, 'high': 1}]
crossover_type = "single_point"
mutation_type = "random"
keep_parents = int(0.05*sol_per_pop)
mutation_percent_genes = 30
stop_criteria="saturate_100"
parent_selection_type="sss"

start_time_iter = time.time()

ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_func,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type = parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        gene_space=gene_space,
                        stop_criteria =stop_criteria)
                        # init_range_low=init_range_low,
                        # init_range_high=init_range_high,

ga_instance.run()
ga_instance.plot_fitness()
print("--- %s seconds ---" % (time.time() - start_time_iter))
solution, solution_fitness, solution_idx = ga_instance.best_solution()

solution = solution.reshape(-1,9)
PBCM_CG_solution = Gen_PBCM_CG_NP(solution,eng,case)
CFD_CG_solution = Gen_CFD_CG(solution,eng,case)
solution = solution.reshape(-1,1,9)
PBCM_CG_solution = PBCM_CG_solution.reshape(1,1,-1)
All_input_solution = np.concatenate((PBCM_CG_solution, solution), axis=2)
CG_predict_solution,_ = predictor_MF_Unet_NB(torch.from_numpy(All_input_solution).to(device='cpu', dtype=torch.float32),MF_Unet,case,device)
CG_predict_solution = CG_predict_solution.detach().numpy()
CG_predict_solution = CG_predict_solution.reshape(-1,100)
xConc = np.linspace(0, 1, num=100)
plt.plot(xConc,Pres_CG.reshape([100,1]), 'k--',xConc, CFD_CG_solution.reshape([100,1]), 'r-',xConc,CG_predict_solution.reshape([100,1]), 'b:')
plt.ylabel("Concentration",fontsize=16)
plt.xlabel("Normalized channel width",fontsize=16)
plt.legend(('Prescribed','Predicted_CFD','Predicted_U-Net'), shadow=True,fontsize=10)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()








#     x = x.reshape(-1,9)
#     PBCM_CG = Gen_PBCM_CG_NP(x,eng,case)
#     x = x.reshape(-1,1,9)
#     PBCM_CG = PBCM_CG.reshape(1,1,-1)
#     All_input = np.concatenate((PBCM_CG, x), axis=2)
#     # start_time_iter = time.time()

#     CG_predict,_ = predictor_MF_Unet_NB(torch.from_numpy(All_input).to(device='cpu', dtype=torch.float32),MF_Unet,case,device)
#     # print("--- %s seconds ---" % (time.time() - start_time_iter))

#     CG_predict = CG_predict.detach().numpy()
#     CG_predict = CG_predict.reshape(-1,100)
#     jd = np.linalg.norm((CG_predict - Pres_CG), ord=1)

#     return jd.item()    

# varbound=np.array([[0,1]]*9)
# start_time = time.time()

# algorithm_param = {'max_num_iteration': 100,\
#                    'population_size':100,\
#                    'mutation_probability':0.1,\
#                    'elit_ratio': 0.01,\
#                    'crossover_probability': 0.5,\
#                    'parents_portion': 0.3,\
#                    'crossover_type':'uniform',\
#                    'max_iteration_without_improv':10}
# model=ga(function=fitness_func,
#          dimension=9,
#          variable_type='real',
#          variable_boundaries=varbound,
#         algorithm_parameters=algorithm_param)

# model.run()
# print("--- %s seconds ---" % (time.time() - start_time))
# convergence=model.report
# solution=model.output_dict






