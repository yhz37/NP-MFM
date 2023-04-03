#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:16:27 2022

@author: junlinou
"""


#%%load module
# import os
#use the specified gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import Extension
from matplotlib import pyplot as plt
import GA
import random
# import Dij
import scipy.io as sio
from numba.cuda.random import create_xoroshiro128p_states
#%%obstacles configuration

##fig. representation
# obstacle = np.float32([[20,40],[15,30],[30,20],[36,25],[35,60],
#                         [30,80],[40,70],[50,75],[40,90],
#                         [52,61],[65,44],[70,58],[67,68],
#                         [80,40],[70,20],[85,15],[92,28],[85,38]])
# num_edge = np.int32([5,4,4,5]).astype(np.int32)
# x_1 = 17
# y_1 = 50
# x_n = 90
# y_n = 40

##fig. 1
obstacle = np.float32([[7,95], [7, 70], [9, 73], [9, 95],
                        [7,55], [7, 40], [9, 40], [9, 58], 
                        [7,25], [7, 7], [27, 7], [27, 9], [9, 9], [9, 25], 
                        [25,40], [25, 20], [27, 20], [27, 37], 
                        [25,65], [25, 53], [27, 50], [27, 65], 
                        [25,95], [25, 80], [27, 80], [27, 95], 
                        [34,56.25], [34, 53.75], [35.5, 52.25], [37, 52.25], [38.5, 53.75], [38.5, 56.25], [37, 57.75], [35.5, 57.75], 
                        [46.5,95], [45, 75], [47, 75], [49, 95], 
                        [44,62], [43, 50], [44, 48], [49, 54], [49, 57], [45, 54], [46, 63], 
                        [40, 43.5], [38, 40], [39, 38], [41, 41.5], 
                        [41,31], [39, 20], [41, 20.5], [43, 31.5], 
                        [38,12], [37, 7], [38.5, 7], [39.5, 12.5], 
                        [56, 95], [56, 93], [71, 93], [71, 95], 
                        [56,80], [56, 60], [58, 61], [58, 79], 
                        [56, 50], [56, 48], [69, 48], [69, 41], [71, 41], [71, 50], 
                        [69,29], [69, 17], [71, 15], [71, 29], 
                        [56,9], [56, 7], [68, 7], [66, 9], 
                        [80,95], [80, 93], [95, 93], [95,95], 
                        [80,80], [80, 57], [82, 55], [82,78], 
                        [80,40], [80, 20], [82, 20], [82,38], 
                        [80,12], [80, 10], [85, 10], [85,12], 
                        [90,12], [90, 10], [95, 10], [95,12]])
num_edge = np.int32([4, 4, 6, 4, 4, 4, 8, 4, 7, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4]).astype(np.int32)
x_1 = 15
y_1 = 30
x_n = 90
y_n = 80


##fig. 2
# obstacle = np.float32([[35,85], [30, 80], [30, 70], [35, 65], [45, 65], [45, 55], [35, 55], [35, 60], [30, 60], [30, 55], [35, 50], [45, 50], [50, 55], [50, 65], [45, 70], [35, 70], [35, 80], [45, 80], [45, 75], [50, 75], [50, 80], [45, 85], [55,50], [50, 45], [50, 20], [55, 15], [65, 15], [70, 20], [70, 25], [65, 25], [65, 20], [55, 20], [55, 45], [65, 45], [65, 40], [70, 40], [70, 45], [65, 50]])
# num_edge = np.int32([22, 16]).astype(np.int32)
# x_1 = 40
# y_1 = 75
# x_n = 57
# y_n = 10

#fig. 3
# obstacle = np.float32([[5,85],[5,5],[95,5],[95,95],[5,95],[5,93],[14,93],[14,80],[35,80],[35,82],[16,82],[16,93],[40,93],[40,65],[70,65],[70,57],[60,57],[60,20],[62,20],[62,55],[75,55],[75,40],[80,40],[80,20],[82,20],[82,42],[77,42],[77,57],[72,57],[72,67],[42,67],[42,93],[93,93],[93,80],[60,80],[60,77],[50,77],[50,75],[62,75],[62,78],[82,78],[82,60],[85,60],[85,50],[87,50],[87,62],[84,62],[84,78],[93,78],[93,7],[75,7],[75,30],[70,30],[70,40],[68,40],[68,28],[73,28],[73,7],[52,7],[52,50],[30,50],[30,70],[28,70],[28,50],[19,50],[19,48],[35,48],[35,16],[15,16],[15,14],[37,14],[37,48],[43,48],[43,15],[45,15],[45,48],[50,48],[50,7],[7,7],[7,30],[20,30],[20,32],[7,32],[7,60],[20,60],[20,62],[7,62],[7,85],
#                         [13,40],[13,38],[26,38],[26,24],[13,24],[13,22],[28,22],[28,40],
#                         [15,73],[15,71],[18,71],[18,68],[20,68],[20,73]])
# num_edge = np.int32([88,8,6]).astype(np.int32)
# x_1 = 12
# y_1 = 11
# x_n = 90
# y_n = 90

##fig. 4
# obstacle = np.float32([[10,90],[10,10],[90,10],[90,12],[12,12],[12,90],
#                         [20,22],[20,20],[90,20],[90,90],[88,90],[88,22],
#                         [20,90],[20,88],[78,88],[78,30],[80,30],[80,90],
#                         [20,80],[20,30],[22,30],[22,78],[70,78],[70,80],
#                         [30,70],[30,30],[70,30],[70,32],[32,32],[32,70],
#                         [40,42],[40,40],[70,40],[70,70],[68,70],[68,42],
#                         [40,70],[40,68],[58,68],[58,50],[60,50],[60,70],
#                         [40,60],[40,50],[42,50],[42,58],[50,58],[50,60]])
# num_edge = np.int32([6,6,6,6,6,6,6,6]).astype(np.int32)
# x_1 = 55
# y_1 = 60.5
# x_n = 5
# y_n = 95


num_generations = 1
times = 1
trend_total = np.zeros((times, num_generations+1)).astype(np.float32)
time_init = np.zeros((times)).astype(np.float32)
time = np.zeros((times)).astype(np.float32)

for k in range(times):
    number_population = 64
    number_candidate = 32
    threads_per_block = (16, 16)#even number
    blocks_per_grid = (int(number_population/threads_per_block[0]), int(number_candidate/threads_per_block[1]))
    #number = number_population*number_candidate
    #generate population
    start = timer()
    L = -1.8
    obstacles = Extension.exten(obstacle,num_edge,L)

    start_point = np.float32([[x_1,y_1]])
    goal_point = np.float32([[x_n,y_n]])
    # L = -1.801
    # obstacles_o = Extension.exten(obstacle,num_edge,L)
    # points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))
    
    obstacles_out = cuda.to_device(obstacles)
    # points_out = cuda.to_device(points)
    # N = points.shape[0]
    # intersect_value = np.ones((N,N)).astype(np.int32)
    # intersect_value_out = cuda.to_device(intersect_value)
    # dist = np.zeros((N,N)).astype(np.float32)
    # dist_out = cuda.to_device(dist)
    num_edge_out = cuda.to_device(num_edge)

    
    # random population
    number_of_genes = 2 * 9
    new_population = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    new_population[:, :, 0] = x_1
    new_population[:, :, 1] = y_1
    new_population[:, :, number_of_genes - 2] = x_n
    new_population[:, :,number_of_genes - 1] = y_n
    new_population_out = cuda.to_device(new_population)
    seed_int = random.randint(1,10000)
    rng_states = create_xoroshiro128p_states(number_population * number_candidate, seed=seed_int)
    # GA.population_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, rng_states)
    GA.popu_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, rng_states)

    #matrix initialization
    intersection_value = np.ones((number_population, number_candidate)).astype(np.int32)
    intersection_value_out = cuda.to_device(intersection_value)
    #popul = np.empty((number, number_of_genes))
    fitness = np.zeros((number_population, number_candidate)).astype(np.float32)
    fitness_value_out = cuda.to_device(fitness)
    fitness_out = cuda.device_array_like(fitness_value_out)
    parents = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    parents_out = cuda.to_device(parents)
    offspring_out = cuda.device_array_like(parents_out)
    
    trend = np.zeros(num_generations+1).astype(np.float32)
    trend_out = cuda.to_device(trend)
    best_individual = np.zeros((number_of_genes)).astype(np.float32)
    best_individual_out = cuda.to_device(best_individual)
    order = np.zeros(1).astype(np.int32)
    order_out = cuda.to_device(order)
                
    length_out = cuda.device_array_like(fitness_value_out)
    smoothness_out = cuda.device_array_like(fitness_value_out)
    new_population= new_population_out.copy_to_host()

    for generation in range(num_generations):
        # checking if the path intersect the obstacles.
        start = timer()
        GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, fitness_value_out)
        cuda.synchronize()
        time_ga = timer() - start
        print('Time taken for 1 is %f seconds.' % time_ga)
        # Selecting the best parents in the population for mating.
        start = timer()
        GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
        #trend_out[generation] = fitness_value_out[0]
        #ime_ga = timer() - start
        
        GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
        cuda.synchronize()
        time_ga = timer() - start
        print('Time taken for 2 is %f seconds.' % time_ga)
        start = timer()
        GA.crossover[blocks_per_grid, (threads_per_block[0], int(threads_per_block[1]/2))](parents_out, offspring_out)
        cuda.synchronize()
        time_ga = timer() - start
        print('Time taken for 3 is %f seconds.' % time_ga)
        # Creating the new population based on the parents and offspring.
        start = timer()
        GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
        cuda.synchronize()
        time_ga = timer() - start
        print('Time taken for 4 is %f seconds.' % time_ga)
        # Adding some variations to the offsrping using mutation.
        start = timer()
        GA.GA_mutation_free[blocks_per_grid, threads_per_block](rng_states, new_population_out, obstacles_out, num_edge_out, generation)
        #GA_tenc.mutation[blocks_per_grid, threads_per_block](random_normal_out, random_int_out, new_population_out, generation)
        cuda.synchronize()
        time_ga = timer() - start
        print('Time taken for 5 is %f seconds.' % time_ga)
        start = timer()
        GA.migration[blocks_per_grid, threads_per_block](new_population_out, parents_out, generation)
        cuda.synchronize()
        time_ga = timer() - start
        print('Time taken for 6 is %f seconds.' % time_ga)



    #

    generation += 1
    GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, fitness_value_out)
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
    GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
    trend = trend_out.copy_to_host()
    trend_total[k,:] = trend
    order = order_out.copy_to_host()
    parents = parents_out.copy_to_host()
    tem = order[0]
    best_individual = parents[tem][0]
    time[k] = timer() - start
            
            


filename = 'GA_GPU_fig_1_GA.mat'
# length, smoothness = Dijk.fitness_path(best_individual)
# print(0.1*length+0.9*smoothness)
# length_short, smoothness_short = Dijk.fitness_path(c[0])
# print(0.1*length_short+0.9*smoothness_short)
# sio.savemat(filename, {'path':best_individual,'length':length,'smoothness':smoothness,'shortest_path':c[0],'length_short':length_short,'smoothness_short':smoothness_short})
# sio.savemat(filename, {'trend':trend_total, 'time':time, 'time_init':time_init})
# sio.savemat(filename, {'total_path':c, 'feasible_path':a, 'add_point_feasible_path':o})

print(trend[num_generations])
print(time)

# plt.figure("obs")
# '''
# n = 0
# for i in range(num_edge.shape[0]):
#     plt.plot([obstacle[n+num_edge[i]-1,0],obstacle[n,0]], [obstacle[n+num_edge[i]-1,1],obstacle[n,1]],c="k")
#     plt.plot(obstacle[n:n+num_edge[i],0], obstacle[n:n+num_edge[i],1],c="k")
#     n += num_edge[i]
# '''
# n = 0
# for i in range(num_edge.shape[0]):
#     plt.plot([obstacles[n+num_edge[i]-1,0],obstacles[n,0]], [obstacles[n+num_edge[i]-1,1],obstacles[n,1]],c="k")
#     plt.plot(obstacles[n:n+num_edge[i],0], obstacles[n:n+num_edge[i],1],c="k")
#     n += num_edge[i]
# #plt.plot(points[path,0], points[path,1],c="k")
# #plt.scatter(free_points[:,0],free_points[:,1],c="k")
# for i in range(len(path)):
#     plt.plot(path[i,range(0,len(path[0]),2)], path[i,range(1,len(path[0]),2)],c="k")




#plt.plot(points[path,0], points[path,1],c="k")
#plt.scatter(free_points[:,0],free_points[:,1],c="k")



plt.figure("optimal path")
n = 0
for i in range(num_edge.shape[0]):
    plt.plot([obstacles[n+num_edge[i]-1,0],obstacles[n,0]], [obstacles[n+num_edge[i]-1,1],obstacles[n,1]],c="k")
    plt.plot(obstacles[n:n+num_edge[i],0], obstacles[n:n+num_edge[i],1],c="k")
    n += num_edge[i]

#for i in range(15):
    #plt.plot(new_population[0,i,range(0,len(path[0]),2)], new_population[0,i,range(1,len(path[0])+1,2)],c="k")
#    plt.scatter(new_population[0,i,range(0,len(path[0]),2)], new_population[0,i,range(1,len(path[0])+1,2)],c="k")
plt.plot(best_individual[range(0,len(best_individual),2)], best_individual[range(1,len(best_individual),2)],c="k")
plt.scatter(best_individual[range(0,len(best_individual),2)], best_individual[range(1,len(best_individual),2)],c="k")

# plt.scatter(c[0][range(0,len(c[0]),2)], c[0][range(1,len(c[0]),2)],c="k")
#plt.plot(parents[0,0,range(0,len(path[0]),2)], parents[0,0,range(1,len(path[0])+1,2)],c="k")
plt.xlim(0, 100)
plt.ylim(0, 100)
