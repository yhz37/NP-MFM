#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:09:03 2022

@author: junlinou
"""


from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


@cuda.jit
def fitness(CG_predict_out, Pres_CG_out, fitness_out):
    x = cuda.grid(1)
    fitness_out[x] = 0
    fitness = 0
    for i in range(CG_predict_out.shape[2]):
        fitness += abs(CG_predict_out[x,0,i]-Pres_CG_out[0,i])
    fitness_out[x] = fitness
    
@cuda.jit('(float32[:, :], float32[:], float32[:], float32[:, :], float32[:], int32)')
def selection(new_population_out, fitness_out, fitness_value_out, parents_out, trend_out, generation):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x = cuda.grid(1)
    summ = 0
    # this is the parallel sorting method: numeration sorting
    for i in range(fitness_out.shape[0]):
        if fitness_out[i] < fitness_out[x] or (fitness_out[i] == fitness_out[x] and i < x):
            summ += 1
    for j in range(new_population_out.shape[1]):
        parents_out[summ][j] = new_population_out[x][j]
    fitness_value_out[summ] = fitness_out[x]
    if summ == 0:
        trend_out[generation] = fitness_out[x]


@cuda.jit('(float32[:, :], float32[:, :])')
def crossover(parents_out, offspring_out):

    # The point at which crossover takes place between two parents. Usually it is at the center.
    x = cuda.grid(1)
    num_offspring = offspring_out.shape[0]
    crossover_point = int(parents_out.shape[1]/2)
    # Index of the first parent to mate.
    parent1_idx = 2 * x%num_offspring
    # Index of the second parent to mate.
    parent2_idx = (2 * x + 1)%num_offspring
    for i in range(crossover_point):
        # The first new offspring will have its first half of its genes taken from the first parent.
        offspring_out[2 * x][i] = parents_out[parent1_idx][i]
        # The first offspring will have its second half of its genes taken from the second parent.
        offspring_out[2 * x][crossover_point + i] = parents_out[parent2_idx][crossover_point + i]
        # The second offspring will have its first half of its genes taken from the first parent.
        offspring_out[2 * x + 1][i] = parents_out[parent2_idx][i]
        # The second offspring will have its second half of its genes taken from the second parent.
        offspring_out[2 * x + 1][crossover_point + i] = parents_out[parent1_idx][crossover_point + i]
    # if number of genes is odd number, the rest of genes needs to be assigned
    offspring_out[2 * x][parents_out.shape[1] - 1] = parents_out[parent2_idx][parents_out.shape[1] - 1]
    offspring_out[2 * x + 1][parents_out.shape[1] - 1] = parents_out[parent1_idx][parents_out.shape[1] - 1]



@cuda.jit('(float32[:, :],  float32[:, :], float32[:, :])')
def new_popul(parents_out, offspring_out, new_population_out):
    x = cuda.grid(1)
    num_offspring = offspring_out.shape[0]
    # 50% of the population would be in the next population
    if x<num_offspring * 0.5:
        for i in range(parents_out.shape[1]):
            new_population_out[x,i] = parents_out[x,i]
    else:
        for i in range(parents_out.shape[1]):
            new_population_out[x,i] = offspring_out[x-int(num_offspring * 0.5),i]


@cuda.jit
def mutation(rng_states, new_population_out):
    # Mutation changes a single gene in each offspring randomly.
    idx = cuda.grid(1)
    # 0.01 is the elite rate
    if idx >= new_population_out.shape[0]*0.01:
        for m in range(new_population_out.shape[1]):
            # 0.25 is the mutation rate
            if xoroshiro128p_uniform_float32(rng_states, idx)<0.25:
                temp = (new_population_out[idx,m] + xoroshiro128p_normal_float32(rng_states, idx))%1
                new_population_out[idx,m] = temp
  

