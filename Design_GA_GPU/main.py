#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:13:53 2022

@author: ouj
"""


import pickle
import matlab.engine
import os
# select the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
# from matplotlib import pyplot as plt
os.chdir('/home/ouj/projects/haizhou/To_Junlin/')
from timeit import default_timer as timer
import GA
import Gen_PBCM_CG_NP as GP
from predictor_MF_Unet_NB import predictor_MF_Unet_NB
import numba_to_pytorch as ntp
from matplotlib import pyplot as plt
import scipy.io as sio

# with open('/home/ouj/projects/haizhou/To_Junlin/MF_Unet2_9_MAP.pkl','rb') as f:  # Python 3: open(..., 'rb')
#      MF_Unet, Final_lr, loss_Train, loss_Val, scheduler,train_set,val_set,case,device = pickle.load(f)
with open('/home/ouj/projects/haizhou/To_Junlin/MF_Unet2_9_MAP_0_1_4000.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    input, PBCM_CG, CFD_CG, MF_Unets, MF_Unet, Final_lr, loss_Train, loss_Val, scheduler, train_set, val_set, case, device = pickle.load(f)


if 'eng' not in locals():
    os.chdir('/home/ouj/projects/haizhou/To_Junlin/Matlab_Code')
    eng = matlab.engine.start_matlab()
    os.chdir('/home/ouj/projects/haizhou/To_Junlin/')

# torch.cuda.set_device(3)

# Pres_CG =  eng.DesignConc_9(matlab.double([0.7,0.8,0.9,1,0,0.8,0.2,0.6,0.4]),100)
Pres_CG =  eng.DesignConc_9(matlab.double([7,8,9,0.8,0.2,0.7,0.3,0.6,0.4]),100)
# Pres_CG =  eng.DesignConc_9(matlab.double([8,8,9,0.1,0.8,0.8,0.8,0.8,0.1]),100)
# Pres_CG =  eng.DesignConc_9(matlab.double([7,8,9,0.3,0.6,0.3,0.3,0.6,0.3]),100)
# Pres_CG =  eng.DesignConc_9(matlab.double([7,8,9,0.9,0.7,0.4,0.1,0.2,0.9]),100)
Pres_CG = np.asarray(Pres_CG).transpose()
Pres_CG_out = cuda.to_device(Pres_CG)
device = 'cuda'
# #data set initialization
# diff = 0
# #termination condition: diff < 0.1 over 20 iterations
# while (diff<0.1):
#     diff += 0.01

#%% initialization for some paremeters
# the number of times the program we run
times = 2
# number of generation
num_generations = 1000
# all the fitness trends for so many times we run the program
trend_total = np.zeros((times, num_generations+1)).astype(np.float32)
# all the runtime for so many times we run the program
time = np.zeros(times).astype(np.float32)
# best individual for each time
best_individuals = np.zeros((times, 9)).astype(np.float32)
for i in range(times):
    # initialization
    start = timer()
    # number of population, candidates, genes, generations
    # number_population is fixed and cannot be changed
    number_population = 1
    number_candidate = 2048
    number_of_genes = 9
    
    # threads of a block and blocks of a grid
    threads_per_block = 64 #even number
    blocks_per_grid = int(number_candidate/threads_per_block)
    # concentration gradient
    conc = np.zeros((number_candidate, 100)).astype(np.float32)
    conc_out = cuda.to_device(conc)
    # population
    new_population = np.random.rand(number_candidate, number_of_genes).astype(np.float32)
    # new_population[:] = [0.3,0.4,0.5,1,0,0.8,0.2,0.6,0.4]
    new_population_out = cuda.to_device(new_population)
    # all input is the integration of concentration gradient and population
    All_input = np.zeros((number_candidate, 100+number_of_genes)).astype(np.float32)
    All_input_out = cuda.to_device(All_input)
    # parents and offsprings
    parents = np.zeros((number_candidate, number_of_genes)).astype(np.float32)
    parents_out = cuda.to_device(parents)
    offspring_out = cuda.device_array_like(parents_out)
    # rng_states is for generating random data
    rng_states = create_xoroshiro128p_states(number_candidate, seed=1)
    base = np.zeros((number_candidate, 100)).astype(np.float32)
    R = np.array([3034506461771.37,10327732726825.7,478026753889.2466]).astype(np.float32)
    R_out = cuda.to_device(R)
    # fitness value
    fitness = np.zeros((number_candidate)).astype(np.float32)
    fitness_out = cuda.to_device(fitness)
    fitness_value_out = cuda.device_array_like(fitness_out)
    # fitness trend
    trend = np.zeros(num_generations+1).astype(np.float32)
    trend_out = cuda.to_device(trend)
    # set initial value for difference and generation
    difference = 1
    generation = 0
    
    while(difference>0.000001):
        # for generation in range(num_generations):
        # caculating the concentration gradient
        # start1 = timer()
        GP.fitness_TripleTMixer[blocks_per_grid, threads_per_block](new_population_out, R_out, conc_out, All_input_out)
        cuda.synchronize()
        # time1 = timer() - start1
        # print(time1)
        All_input_t = ntp.devndarray2torch(All_input_out)
        
        #GA.fitness[blocks_per_grid, threads_per_block](new_population_out, R_out, conc_out)
        # conc = conc_out.copy_to_host()
        # new_population = new_population_out.copy_to_host()
        # All_input = np.concatenate((conc, new_population), axis=1)#torch.concatenate
        # start1 = timer()
        # get the prediction about the concentration
        CG_predict,_ = predictor_MF_Unet_NB(torch.reshape(All_input_t, (number_candidate,1,109)),MF_Unet,case,device)
        # time1 = timer() - start1
        # print(time1)
        #CG_predict,_ = predictor_MF_Unet_NB(torch.from_numpy(All_input.reshape((number_candidate,1,109))).to(device='cuda', dtype=torch.float32),MF_Unet,case,device)
        CG_predict_out = cuda.as_cuda_array(CG_predict)
        # calculate the fitness
        GA.fitness[blocks_per_grid, threads_per_block](CG_predict_out, Pres_CG_out, fitness_out)
        # CG_predict = CG_predict.detach().numpy()
        # CG_predict = CG_predict.reshape(-1,100)
        # fitness = np.linalg.norm((CG_predict - Pres_CG), ord=1, axis=1).astype(np.float32)
        # fitness_out = cuda.to_device(fitness)
        # selection in genetic algorithm
        GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_out, fitness_value_out, parents_out, trend_out, generation)
        cuda.synchronize()
        trend = trend_out.copy_to_host()
        if generation >= 50:
            # I think the average relative change equal to the relative change devided by 50
            difference = (trend[generation-50] - trend[generation])/50
        # crossover process in genetic algorithm
        GA.crossover[blocks_per_grid, int(threads_per_block/2)](parents_out, offspring_out)
        cuda.synchronize()
        # get the new population after crossover
        GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
        cuda.synchronize()
        # mutation process in genetic algorithm
        GA.mutation[blocks_per_grid, threads_per_block](rng_states, new_population_out)
        cuda.synchronize()
        generation += 1
    #generation += 1
    GP.fitness_TripleTMixer[blocks_per_grid, threads_per_block](new_population_out, R_out, conc_out, All_input_out)
    cuda.synchronize()
    All_input_t = ntp.devndarray2torch(All_input_out)
    # get the prediction about the concentration
    CG_predict,_ = predictor_MF_Unet_NB(torch.reshape(All_input_t, (number_candidate,1,109)),MF_Unet,case,device)
    CG_predict_out = cuda.as_cuda_array(CG_predict)
    # calculate the fitness
    GA.fitness[blocks_per_grid, threads_per_block](CG_predict_out, Pres_CG_out, fitness_out)
    # selection in genetic algorithm
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_out, fitness_value_out, parents_out, trend_out, generation)
    trend = trend_out.copy_to_host()
    fitness_value = fitness_value_out.copy_to_host()
    fitness = fitness_out.copy_to_host()
    parents = parents_out.copy_to_host()
    best_individuals[i] = parents[0]
    time[i] = timer() - start
    trend_total[i] =trend
    plt.figure("comparison")
    plt.plot(Pres_CG[0,:])
    plt.plot(CG_predict.cpu().detach().numpy()[0,0,:])
    plt.show()
    
filename = 'data.mat'
sio.savemat(filename, {'trend':trend_total, 'time':time, 'best_individuals':best_individuals})
    # print(time)
