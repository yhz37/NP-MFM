#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:25:07 2022

@author: ouj, seonghyeonhong
"""


import math
import numpy as np
from numba import cuda, float32, jit
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32



@cuda.jit    
def fitness_TripleTMixer(pop, R, conc, All_input_out):
# def fitness_TripleTMixer(pop, fitness, alp_des, R, mZ, mS, mSsc, mYsc, mgamma, mbeta, mtheta, pbcm_mat, cfd_mat):     
    x, y = cuda.grid(2)         
    Di = 1e-10
    u = 1e-6    
    h = 60*u        
    Vout = 0
    fnum = 151
    
    C = cuda.local.array((6), dtype=float32)
    V = cuda.local.array((10), dtype=float32)
    Vr1 = pop[x,y,0]
    Vr2 = pop[x,y,1]
    Vr3 = pop[x,y,2]
    C[0] = pop[x,y,3]
    C[1] = pop[x,y,4]
    C[2] = pop[x,y,5]
    C[3] = pop[x,y,6]
    C[4] = pop[x,y,7]
    C[5] = pop[x,y,8]
    
    dP1 = -math.log(1/(0.63384176*Vr1+0.36602540)-1)/0.01373265 + 50
    dP2 = -math.log(1/(0.63384176*Vr2+0.36602540)-1)/0.01373265 + 50
    dP3 = -math.log(1/(0.63384176*Vr3+0.36602540)-1)/0.01373265 + 50
    
    V[9] = R[2]*(dP1/R[1] + dP2/R[1] + dP3/R[1])
    
    V[6] = V[9] + dP1
    V[7] = V[9] + dP2
    V[8] = V[9] + dP3
    
    V[0] = V[6] + dP1/(R[1]*(2/R[0]))
    V[1] = V[0]
    V[2] = V[7] + dP2/(R[1]*(2/R[0]))
    V[3] = V[2]
    V[4] = V[8] + dP3/(R[1]*(2/R[0]))
    V[5] = V[4]

    
    Q = cuda.local.array((10), dtype=float32)     
    Q[0] = (V[0]-V[6])/R[0]
    Q[1] = (V[1]-V[6])/R[0]
    Q[2] = (V[2]-V[7])/R[0]
    Q[3] = (V[3]-V[7])/R[0]
    Q[4] = (V[4]-V[8])/R[0]
    Q[5] = (V[5]-V[8])/R[0]
    Q[6] = (V[6]-V[9])/R[1]
    Q[7] = (V[7]-V[9])/R[1]
    Q[8] = (V[8]-V[9])/R[1]
    Q[9] = (V[9]-Vout)/R[2]

            
    # 1st Layer                    
    Am1 = cuda.local.array((3, 1), dtype=float32)
    Am2 = cuda.local.array((3, 1), dtype=float32)
    Bm = cuda.local.array((3, 151), dtype=float32)
    Cm = cuda.local.array((2, 151), dtype=float32)        
        
    Am1[0,0] = C[0]
    Am2[0,0] = C[1]
    Am1[1,0] = C[2]
    Am2[1,0] = C[3]
    Am1[2,0] = C[4]
    Am2[2,0] = C[5]
    
    s1 = Q[0]/(Q[0]+Q[1])
    s2 = Q[2]/(Q[2]+Q[3])
    s3 = Q[4]/(Q[4]+Q[5])
    
    Bm[0,0] = Am1[0,0]*s1 + Am2[0,0]*(1-s1)
    Bm[1,0] = Am1[1,0]*s2 + Am2[1,0]*(1-s2)
    Bm[2,0] = Am1[2,0]*s3 + Am2[2,0]*(1-s3) 
       
    for n in range(1,fnum):
        sum1 = 0
        sum2 = 0
        sum3 = 0
                
        for m in range(1):
            
            f1 = (m-n*s1)*math.pi
            f2 = (m+n*s1)*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum1 += Am1[0,m]*( math.sin(2*n*math.pi*s1)/(2*n*math.pi) + s1 )
            else:
                sum1 += Am1[0,m]*s1*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)
                            
            f1 = (m-n*s2)*math.pi
            f2 = (m+n*s2)*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum2 += Am1[1,m]*( math.sin(2*n*math.pi*s2)/(2*n*math.pi) + s2 )
            else:
                sum2 += Am1[1,m]*s2*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)                
            
            f1 = (m-n*s3)*math.pi
            f2 = (m+n*s3)*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum3 += Am1[2,m]*( math.sin(2*n*math.pi*s3)/(2*n*math.pi) + s3 )
            else:
                sum3 += Am1[2,m]*s3*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)
                              
            F1 = (m+n-n*s1)*math.pi
            F2 = (m-n+n*s1)*math.pi
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum1 += Am2[0,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s1-1))/(n*math.pi) + math.cos(n*math.pi*s1)*(1-s1) )
            else:
                sum1 += Am2[0,m]*2*(-1)**n*(1-s1)*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )
                            
            F1 = (m+n-n*s2)*math.pi
            F2 = (m-n+n*s2)*math.pi
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum2 += Am2[1,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s2-1))/(n*math.pi) + math.cos(n*math.pi*s2)*(1-s2) )
            else:
                sum2 += Am2[1,m]*2*(-1)**n*(1-s2)*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )                
            
            F1 = (m+n-n*s3)*math.pi
            F2 = (m-n+n*s3)*math.pi  
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum3 += Am2[2,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s3-1))/(n*math.pi) + math.cos(n*math.pi*s3)*(1-s3) )
            else:
                sum3 += Am2[2,m]*2*(-1)**n*(1-s3)*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )
            
        Bm[0,n] = sum1
        Bm[1,n] = sum2
        Bm[2,n] = sum3
    
    # 2nd Layer    
    L2 = 51239.342*u
    w2 = 313.44465*u  
    a2 = w2*h                    
    for n in range(fnum):
        Bm[0,n] = Bm[0,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[6]/a2*w2/Di) )
        Bm[1,n] = Bm[1,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[7]/a2*w2/Di) )
        Bm[2,n] = Bm[2,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[8]/a2*w2/Di) )
    
    # Psi Mixer           
    s = Q[6] / (Q[6] + Q[7])        
    Cm[0,0] = Bm[0,0]*s + Bm[1,0]*(1-s)    
    for n in range(1,fnum):        
        sum1, sum2 = 0, 0            
        f1, f2 = 0, 0
        F1, F2 = 0, 0
        for m in range(fnum):            
            f1 = (m-n*s)*math.pi
            f2 = (m+n*s)*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum1 += Bm[0,m]*( math.sin(2*n*math.pi*s)/(2*n*math.pi) + s )
            else:
                sum1 += Bm[0,m]*s*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)                
            F1 = (m+n-n*s)*math.pi
            F2 = (m-n+n*s)*math.pi
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum2 += Bm[1,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s-1))/(n*math.pi) + math.cos(n*math.pi*s)*(1-s) )
            else:
                sum2 += Bm[1,m]*2*(-1)**n*(1-s)*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )            
        Cm[0,n] = sum1 + sum2

    s = (Q[6] + Q[7]) / (Q[6] + Q[7] + Q[8])    
    Cm[1,0] = Cm[0,0]*s + Bm[2,0]*(1-s)
    for n in range(1,fnum):        
        sum1, sum2 = 0, 0            
        f1, f2 = 0, 0
        F1, F2 = 0, 0       
        for m in range(fnum):            
            f1 = (m-n*s)*math.pi
            f2 = (m+n*s)*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum1 += Cm[0,m]*( math.sin(2*n*math.pi*s)/(2*n*math.pi) + s )
            else:
                sum1 += Cm[0,m]*s*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)                
            F1 = (m+n-n*s)*math.pi
            F2 = (m-n+n*s)*math.pi
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum2 += Bm[2,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s-1))/(n*math.pi) + math.cos(n*math.pi*s)*(1-s) )
            else:
                sum2 += Bm[2,m]*2*(-1)**n*(1-s)*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )            
        Cm[1,n] = sum1 + sum2
    
    # 3rd Layer
    dloc = 400*u
    w3 = 1200*u
    a3 = w3*h                    
    for n in range(fnum):
        Cm[0,n] = Cm[1,n]*math.exp(-(n*math.pi)**2*dloc/(w3*Q[9]/a3*w3/Di) )      
    
    for m in range(100):
        conc[x,y,m] = 0
        for n in range(fnum):
            conc[x,y,m] += Cm[0,n]*math.cos(n*math.pi*1/99*m)
        All_input_out[x,y,m] = conc[x,y,m]
    for m in range(9):
        All_input_out[x,y,100+m] = pop[x,y,m]


@cuda.jit
def fitness_old(CG_predict_out, Pres_CG_out, fitness_out):
    x, y = cuda.grid(2)
    fitness_out[x,y] = 0
    fitness = 0
    for i in range(CG_predict_out.shape[2]):
        fitness += abs(CG_predict_out[x,y,i]-Pres_CG_out[0,i])
    fitness_out[x,y] = fitness


@cuda.jit
def fitness(CG_predict_out, Pres_CG_out, fitness_out):
    x, y = cuda.grid(2)
    fitness_out[x,y] = 0
    fitness = 0
    for i in range(CG_predict_out.shape[2]):
        fitness += (CG_predict_out[x,y,i]-Pres_CG_out[0,i])**2
    fitness_out[x,y] = math.sqrt(fitness)

@cuda.jit('(float32[:, :, :], float32[:, :], float32[:, :], float32[:, :, :])')
def rank1(new_population_out, fitness_out, fitness_value_out, parents_out):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x, y = cuda.grid(2)
    summ = 0
    # this is the parallel sorting method: numeration sorting
    for i in range(fitness_out.shape[1]):
        if fitness_out[x,i] < fitness_out[x,y] or (fitness_out[x,i] == fitness_out[x,y] and i < y):
            summ += 1
    for j in range(new_population_out.shape[2]):
        parents_out[x,summ,j] = new_population_out[x,y,j]
    fitness_value_out[x,summ] = fitness_out[x,y]

@cuda.jit('(float32[:, :], float32[:], int32[:])')
def rank2(fitness_value_out, best_fitness_out, order):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x = cuda.grid(1)
    summ = 0

    for i in range(fitness_value_out.shape[0]):
        if fitness_value_out[i,0] < fitness_value_out[x,0] or (fitness_value_out[i,0] == fitness_value_out[x,0] and i < x):
            summ += 1
    if summ == 0:
        best_fitness_out[0] = fitness_value_out[x][0]
        order[0] = x

# Rank Population by Fitness
@cuda.jit('(float32[:, :, :], float32[:, :,:], float32[:, :])')
def rank_population(pop, rankpop, fitness):    
    x, y = cuda.grid(2)
    summ = 0
    for i in range(rankpop.shape[1]):
        if fitness[x,i]<fitness[x,y] or (fitness[x,i]==fitness[x,y] and i<y):
            summ += 1
    for j in range(rankpop.shape[2]):
        rankpop[x,summ,j] = pop[x,y,j]


@cuda.jit('(float32[:, :, :], float32[:, :, :])')
def migration(pop, rankpop):
    # 2% of the best individuals would be transferred to next population
    x, y = cuda.grid(2)    
    # Index of the next population
    if y < int(0.02 * pop.shape[1]):
        pop_idx = (x + 1)%rankpop.shape[0]
        for i in range(pop.shape[2]):
            pop[x,y,i] = rankpop[pop_idx,y,i]
            

# Truncation Selection
@cuda.jit
def crossover_truncation(rng_states, rankpop, offspring):

    x, y = cuda.grid(2)    
    T = round(0.15*rankpop.shape[2])
    P1 = round(xoroshiro128p_uniform_float32(rng_states, y)*T)    
    P2 = round(xoroshiro128p_uniform_float32(rng_states, y)*T)    
    
    crossover_point = 3 # crossover between deltaP and 6 concentrations
    cprob = xoroshiro128p_uniform_float32(rng_states, y)
    
    if cprob <= 0.85:
        for i in range(crossover_point):
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[x,y,i] = rankpop[x,P1,i]
        for i in range(crossover_point, rankpop.shape[2]):
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[x,y,i] = rankpop[x,P2,i]    
    else:
        for i in range(rankpop.shape[2]):
            offspring[x,y,i] = rankpop[x,P1,i]
    
    

# Tournament Selection
@cuda.jit
def crossover_tournament(rng_states, rankpop, offspring, K):    
    
    x, y = cuda.grid(2)    
    P1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(rankpop.shape[1]))
    P2 = round(xoroshiro128p_uniform_float32(rng_states, y)*(rankpop.shape[1]))    
    
    for i in range(K):
        temp1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(rankpop.shape[1]))
        temp2 = round(xoroshiro128p_uniform_float32(rng_states, y)*(rankpop.shape[1]))
        if temp1 < P1:
            P1 = temp1
        if temp2 < P2:
            P2 = temp2
            
    crossover_point = 3 # crossover between deltaP and 6 concentrations
    cprob = xoroshiro128p_uniform_float32(rng_states, y)
    
    if cprob <= 0.85:
        for i in range(crossover_point):
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[x,y,i] = rankpop[x,P1,i]
        for i in range(crossover_point, rankpop.shape[2]):
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[x,y,i] = rankpop[x,P2,i]    
    else:
        for i in range(rankpop.shape[2]):
            offspring[x,y,i] = rankpop[x,P1,i]


# Mutation
@cuda.jit
def mutation_random(rng_states, pop, rankpop, offspring):
    mrate = 0.05
    # Mutation changes a single gene in each offspring randomly.
    x, y = cuda.grid(2)    
    
    # 2% of individuals - elite
    if y >= round(pop.shape[1]*0.02): # 2% of individuals
        for i in range(pop.shape[2]):
            pop[x,y,i] = offspring[x,y,i]
    else:
        for i in range(pop.shape[2]):
            pop[x,y,i] = rankpop[x,y,i]
    
    # only mutate 25% of non-elite    
    # mutate ration term
    muprob = xoroshiro128p_uniform_float32(rng_states, y)
    if muprob <= 0.25 and y >= round(pop.shape[1]*0.02): 
        x1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(2))
        pop[x,y,x1] = (pop[x,y,x1] + xoroshiro128p_normal_float32(rng_states, y)*mrate*1)%1
        
                             
    muprob = xoroshiro128p_uniform_float32(rng_states, y)
    if muprob <= 0.25 and y >= round(pop.shape[1]*0.02): 
    
        x1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(2)) + 3
        x2 = round(xoroshiro128p_uniform_float32(rng_states, y)*(2)) + 6
        pop[x,y,x1] = (pop[x,y,x1] + xoroshiro128p_normal_float32(rng_states, y)*mrate*1)%1
        pop[x,y,x2] = (pop[x,y,x2] + xoroshiro128p_normal_float32(rng_states, y)*mrate*1)%1

@cuda.jit
def test_random(rng_states, test_out):    
    
    x, y = cuda.grid(2)    
    test_out[x,y,0] = xoroshiro128p_uniform_float32(rng_states, y)
    test_out[x,y,1] = xoroshiro128p_uniform_float32(rng_states, y)
    test_out[x,y,2] = xoroshiro128p_uniform_float32(rng_states, y)
