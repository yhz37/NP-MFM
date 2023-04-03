#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 08:06:12 2020
    single point crossover
    Gaussian mutation
    truncation selection / Tournement Selection
    migration
@author: seonghyeonhong
"""

#%% Load Modules

import numpy as np
import math
import matplotlib.pyplot as plt

from numba import cuda, float32
from timeit import default_timer as timer
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32


# cuda.close()
cuda.select_device(1)


#%% Define Parameters

# Micrometer
u = 1e-6

# Channel Depth
h = 60*u

# Viscosity
mu = 0.001

# Diffusivity
Di = 1e-10

# Order of Fourier Coefficient
fnum = 100

# Detection Location
dloc = 400*u

#%% Defined Functions

def TripleTmixer(V, C, R):
    
    Di = 1e-10
    u = 1e-6    
    h = 60*u    
    
    Vout = 0    
    
    Q = np.zeros((10,))        
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
    
    flag = 0
    for ff in range(len(Q)):
        if Q[ff]<0:
            flag += 1
    
    # 1st Layer
    
    s = np.zeros((3))    
    s[0] = Q[0]/(Q[0]+Q[1])
    s[1] = Q[2]/(Q[2]+Q[3])
    s[2] = Q[4]/(Q[4]+Q[5])
            
    Am1 = np.zeros((3,fnum))
    Am2 = np.zeros((3,fnum))
    Bm = np.zeros((3,fnum))
    Cm = np.zeros((2,fnum))
    Dm = np.zeros((1,fnum))
        
    Am1[0,0] = C[0]
    Am2[0,0] = C[1]
    Am1[1,0] = C[2]
    Am2[1,0] = C[3]
    Am1[2,0] = C[4]
    Am2[2,0] = C[5]
    
    Bm[0,0] = Am1[0,0]*s[0] + Am2[0,0]*(1-s[0])
    Bm[1,0] = Am1[1,0]*s[1] + Am2[1,0]*(1-s[1])
    Bm[2,0] = Am1[2,0]*s[2] + Am2[2,0]*(1-s[2])
    
    for n in range(1,fnum):
        sum1 = np.zeros((3))
        sum2 = np.zeros((3))
        
        for m in range(1):
            
            f1 = (m-n*s)*math.pi
            f2 = (m+n*s)*math.pi
            
            if abs(f1[0])<1e-20 or abs(f2[0])<1e-20:
                sum1[0] += Am1[0,m]*( math.sin(2*n*math.pi*s[0])/(2*n*math.pi) + s[0] )
            else:
                sum1[0] += Am1[0,m]*s[0]*( f1[0]*math.sin(f2[0]) + f2[0]*math.sin(f1[0]) )/(f1[0]*f2[0])
                
            if abs(f1[1])<1e-20 or abs(f2[1])<1e-20:
                sum1[1] += Am1[1,m]*( math.sin(2*n*math.pi*s[1])/(2*n*math.pi) + s[1] )
            else:
                sum1[1] += Am1[1,m]*s[1]*( f1[1]*math.sin(f2[1]) + f2[1]*math.sin(f1[1]) )/(f1[1]*f2[1])
                
            if abs(f1[2])<1e-20 or abs(f2[2])<1e-20:
                sum1[2] += Am1[2,m]*( math.sin(2*n*math.pi*s[2])/(2*n*math.pi) + s[2] )
            else:
                sum1[2] += Am1[2,m]*s[2]*( f1[2]*math.sin(f2[2]) + f2[2]*math.sin(f1[2]) )/(f1[2]*f2[2])
                
            F1 = (m+n-n*s)*math.pi
            F2 = (m-n+n*s)*math.pi
            
            if abs(F1[0])<1e-20 or abs(F2[0])<1e-20:
                sum2[0] += Am2[0,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[0]-1))/(n*math.pi) + math.cos(n*math.pi*s[0])*(1-s[0]) )
            else:
                sum2[0] += Am2[0,m]*2*(-1)**n*(1-s[0])*( math.cos(F2[0]/2)*math.sin(F1[0]/2)/F1[0] + math.cos(F1[0]/2)*math.sin(F2[0]/2)/F2[0] )
            
            if abs(F1[1])<1e-20 or abs(F2[1])<1e-20:
                sum2[1] += Am2[1,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[1]-1))/(n*math.pi) + math.cos(n*math.pi*s[1])*(1-s[1]) )
            else:
                sum2[1] += Am2[1,m]*2*(-1)**n*(1-s[1])*( math.cos(F2[1]/2)*math.sin(F1[1]/2)/F1[1] + math.cos(F1[1]/2)*math.sin(F2[1]/2)/F2[1] )
                
            if abs(F1[2])<1e-20 or abs(F2[2])<1e-20:
                sum2[2] += Am2[2,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[2]-1))/(n*math.pi) + math.cos(n*math.pi*s[2])*(1-s[2]) )
            else:
                sum2[2] += Am2[2,m]*2*(-1)**n*(1-s[2])*( math.cos(F2[2]/2)*math.sin(F1[2]/2)/F1[2] + math.cos(F1[2]/2)*math.sin(F2[2]/2)/F2[2] )
            
        Bm[0,n] = sum1[0] + sum2[0]
        Bm[1,n] = sum1[1] + sum2[1]
        Bm[2,n] = sum1[2] + sum2[2]
    
    
    # 2nd Layer    
    L2 = 51239.342*u
    w2 = 313.44465*u  
    a2 = w2*h                    
    for n in range(fnum):
        Bm[0,n] = Bm[0,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[6]/a2*w2/Di) )
        Bm[1,n] = Bm[1,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[7]/a2*w2/Di) )
        Bm[2,n] = Bm[2,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[8]/a2*w2/Di) )
    
    # Psi Mixer
    s = np.zeros((2))
    s[0] = Q[6] / (Q[6] + Q[7])
    
    
    Cm[0,0] = Bm[0,0]*s[0] + Bm[1,0]*(1-s[0])    
    for n in range(1,fnum):        
        sum1 = 0
        sum2 = 0        
        for m in range(fnum):            
            f1 = (m-n*s[0])*math.pi
            f2 = (m+n*s[0])*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum1 += Bm[0,m]*( math.sin(2*n*math.pi*s[0])/(2*n*math.pi) + s[0] )
            else:
                sum1 += Bm[0,m]*s[0]*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)                
                
            F1 = (m+n-n*s[0])*math.pi
            F2 = (m-n+n*s[0])*math.pi
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum2 += Bm[1,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[0]-1))/(n*math.pi) + math.cos(n*math.pi*s[0])*(1-s[0]) )
            else:
                sum2 += Bm[1,m]*2*(-1)**n*(1-s[0])*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )            
        Cm[0,n] = sum1 + sum2

    s[1] = (Q[6] + Q[7]) / (Q[6] + Q[7] + Q[8])    
    Cm[1,0] = Cm[0,0]*s[1] + Bm[2,0]*(1-s[1])
    for n in range(1,fnum):        
        sum1 = 0
        sum2 = 0        
        for m in range(fnum):            
            f1 = (m-n*s[1])*math.pi
            f2 = (m+n*s[1])*math.pi
            if abs(f1)<1e-20 or abs(f2)<1e-20:
                sum1 += Cm[0,m]*( math.sin(2*n*math.pi*s[1])/(2*n*math.pi) + s[1] )
            else:
                sum1 += Cm[0,m]*s[1]*( f1*math.sin(f2) + f2*math.sin(f1) )/(f1*f2)                
            F1 = (m+n-n*s[1])*math.pi
            F2 = (m-n+n*s[1])*math.pi
            if abs(F1)<1e-20 or abs(F2)<1e-20:
                sum2 += Bm[2,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[1]-1))/(n*math.pi) + math.cos(n*math.pi*s[1])*(1-s[1]) )
            else:
                sum2 += Bm[2,m]*2*(-1)**n*(1-s[1])*( math.cos(F2/2)*math.sin(F1/2)/F1 + math.cos(F1/2)*math.sin(F2/2)/F2 )            
        Cm[1,n] = sum1 + sum2
    
    # 3rd Layer
    dloc = 400*u
    w3 = 1200*u
    a3 = w3*h                    
    for n in range(fnum):
        Dm[0,n] = Cm[1,n]*math.exp(-(n*math.pi)**2*dloc/(w3*Q[9]/a3*w3/Di) )               
    
    # return Dm
    return Dm[0,:]


#%% Tmixer Simulation

# Channel Geometry

# Channel Layer 1
L1 = 4000*u
w1 = 110.82*u
b1 = w1/h
a1 = w1*h

# Channel Layer 2
L2 = 51239.342*u
w2 = 313.44465*u
b2 = w2/h
a2 = w2*h

# Channel Layer 3
L3 = 10000*u
w3 = 1200*u
b3 = w3/h
a3 = w3*h

# Compute Resistance for different layers
T1, T2, T3 = 0, 0, 0
for n in range(1,fnum+1,2):
    T1 += math.tanh(n*math.pi/(2*b1))/(n**5) 
    T2 += math.tanh(n*math.pi/(2*b2))/(n**5) 
    T3 += math.tanh(n*math.pi/(2*b3))/(n**5) 
R1 = 12*mu*b1*L1/(w1**4)/(1-(192/math.pi**5)*b1*T1)
R2 = 12*mu*b2*L2/(w2**4)/(1-(192/math.pi**5)*b2*T2)
R3 = 12*mu*b3*L3/(w3**4)/(1-(192/math.pi**5)*b3*T3)
R = np.array([R1, R2, R3])

V = np.zeros((10,))
V[0] = 500
V[1] = 500
V[2] = 500
V[3] = 500
V[4] = 500
V[5] = 500
Vout = 0

A1 = R2/R1 + R2/R1 + 1
A2 = R2/R1 + R2/R1 + 1
A3 = R2/R1 + R2/R1 + 1
A4 = R3/R2 + R3/R2 + R3/R2 + 1
D1 = V[0]*R2/R1 + V[1]*R2/R1
D2 = V[2]*R2/R1 + V[3]*R2/R1
D3 = V[4]*R2/R1 + V[5]*R2/R1

alpha = R3/R2*A2*A3 + R3/R2*A1*A3 + R3/R2*A1*A2 - A1*A2*A3*A4
beta = -R3/R2*D1*A2*A3 -R3/R2*D2*A1*A3 - R3/R2*D3*A1*A2
V16 = beta/alpha

V12 = (D1 + V16)/A1
V34 = (D2 + V16)/A2
V56 = (D3 + V16)/A3

Am1 = np.zeros(fnum)
Am2 = np.zeros(fnum)
Bm = np.zeros(fnum)

V[6] = V12
V[7] = V34
V[8] = V56
V[9] = V16

# Initial Concentrations
C = np.zeros((6,))
C[0] = 0.9
C[1] = 0.1
C[2] = 0.8
C[3] = 0.2
C[4] = 0.7
C[5] = 0.3

Dm = TripleTmixer(V, C, R)

et = np.linspace(0, 1, 101)
descon = np.zeros(len(et))

for m in range(len(et)):
    for n in range(fnum):
        descon[m] += Dm[n]*math.cos(n*math.pi*et[m])        


#%% GA functions

# Initial Population
def population(num_pop, num_ind, num_gene, upper_limit, lower_limit):
        
    np.random.seed(1)
    pop = np.random.rand(num_pop, num_ind, num_gene)
    pop = pop*(upper_limit-lower_limit) + lower_limit
    
    return pop.astype(np.float32)


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
            

# Selection
@cuda.jit
def crossover_truncation(rng_states, rankpop, offspring, tratio):

    x, y = cuda.grid(2)    
    T = round(tratio*rankpop.shape[2])
    P1 = round(xoroshiro128p_uniform_float32(rng_states, y)*T)    
    P2 = round(xoroshiro128p_uniform_float32(rng_states, y)*T)    
    
    crossover_point = rankpop.shape[2]//2
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
            
    crossover_point = rankpop.shape[2]//2
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
def mutation_random(rng_states, pop, rankpop, offspring, mrate):
    
    # Mutation changes a single gene in each offspring randomly.
    x, y = cuda.grid(2)    
    
    # 2% of individuals - elite
    if y >= round(pop.shape[1]*0.02): # 2% of individuals
        for i in range(pop.shape[2]):
            pop[x,y,i] = offspring[x,y,i]
    else:
        for i in range(pop.shape[2]):
            pop[x,y,i] = rankpop[x,y,i]
    
    muprob = xoroshiro128p_uniform_float32(rng_states, y)
        
    # only mutate 25% of non-elite
    if muprob <= 0.25 and y >= round(pop.shape[1]*0.02): 
    
        x1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(pop.shape[2]//4-1))
        x2 = round(xoroshiro128p_uniform_float32(rng_states, y)*(pop.shape[2]//4-1)) + pop.shape[2]//4
        x3 = round(xoroshiro128p_uniform_float32(rng_states, y)*(pop.shape[2]//4-1)) + (pop.shape[2]//4)*2
        x4 = round(xoroshiro128p_uniform_float32(rng_states, y)*(pop.shape[2]//4-1)) + (pop.shape[2]//4)*3
        pop[x,y,x1] += xoroshiro128p_normal_float32(rng_states, y)*mrate*1
        pop[x,y,x2] += xoroshiro128p_normal_float32(rng_states, y)*mrate*1
        pop[x,y,x3] += xoroshiro128p_normal_float32(rng_states, y)*mrate*1000
        pop[x,y,x4] += xoroshiro128p_normal_float32(rng_states, y)*mrate*1000
        
        # Lower and Upper Bounds
        for i in range(pop.shape[2]):
            if pop[x,y,i] < 0:
                pop[x,y,i] = 0
            elif pop[x,y,i] > 1 and i < pop.shape[2]//2:
                pop[x,y,i] = 1
            elif pop[x,y,i] > 1000 and i >= pop.shape[2]//2:
                pop[x,y,i] = 1000    
                             

@cuda.jit    
def fitness_calculation(pop, fitness, descon, R, et):
     
    x, y = cuda.grid(2)         
    penalty = 1000    
    Di = 1e-10
    u = 1e-6    
    h = 60*u        
    Vout = 0
    fnum = 100
    
    C = cuda.local.array((6), dtype=float32)
    V = cuda.local.array((10), dtype=float32)
    C[0] = pop[x,y,0]
    C[1] = pop[x,y,1]
    C[2] = pop[x,y,2]
    C[3] = pop[x,y,3]
    C[4] = pop[x,y,4]
    C[5] = pop[x,y,5]
    V[0] = pop[x,y,6]
    V[1] = pop[x,y,7]
    V[2] = pop[x,y,8]
    V[3] = pop[x,y,9]
    V[4] = pop[x,y,10]
    V[5] = pop[x,y,11]
    # for n in range(6):
    #     C[n] = pop[x,y,n]
    #     V[n] = pop[x,y,n+6]
    
    # Compute Node Pressures
    A1 = R[1]/R[0] + R[1]/R[0] + 1
    A2 = R[1]/R[0] + R[1]/R[0] + 1
    A3 = R[1]/R[0] + R[1]/R[0] + 1
    A4 = R[2]/R[1] + R[2]/R[1] + R[2]/R[1] + 1
    D1 = V[0]*R[1]/R[0] + V[1]*R[1]/R[0]
    D2 = V[2]*R[1]/R[0] + V[3]*R[1]/R[0]
    D3 = V[4]*R[1]/R[0] + V[5]*R[1]/R[0]
    
    alpha = R[2]/R[1]*(A2*A3 + A1*A3 + A1*A2) - A1*A2*A3*A4
    beta = -R[2]/R[1]*(D1*A2*A3 + D2*A1*A3 + D3*A1*A2)
    V[9] = beta/alpha    
    V[6] = (D1 + V[9])/A1
    V[7] = (D2 + V[9])/A2
    V[8] = (D3 + V[9])/A3
    
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

    flag = 0
    for ff in range(len(Q)):
        if Q[ff] < 0:
            flag += 1
            
    if flag > 0:
        fitness[x,y] = penalty
        
    else:                
        # 1st Layer
        s = cuda.local.array((3), dtype=float32)
        s[0] = Q[0]/(Q[0]+Q[1])
        s[1] = Q[2]/(Q[2]+Q[3])
        s[2] = Q[4]/(Q[4]+Q[5])
                
        Am1 = cuda.local.array((3, 1), dtype=float32)
        Am2 = cuda.local.array((3, 1), dtype=float32)
        Bm = cuda.local.array((3, 100), dtype=float32)
        Cm = cuda.local.array((2, 100), dtype=float32)        
            
        Am1[0,0] = C[0]
        Am2[0,0] = C[1]
        Am1[1,0] = C[2]
        Am2[1,0] = C[3]
        Am1[2,0] = C[4]
        Am2[2,0] = C[5]
        
        Bm[0,0] = Am1[0,0]*s[0] + Am2[0,0]*(1-s[0])
        Bm[1,0] = Am1[1,0]*s[1] + Am2[1,0]*(1-s[1])
        Bm[2,0] = Am1[2,0]*s[2] + Am2[2,0]*(1-s[2]) 
   
        sum1 = cuda.local.array((3), dtype=float32)     
        sum2 = cuda.local.array((3), dtype=float32)     
        f1 = cuda.local.array((3), dtype=float32)     
        f2 = cuda.local.array((3), dtype=float32)     
        F1 = cuda.local.array((3), dtype=float32)     
        F2 = cuda.local.array((3), dtype=float32)   
        for n in range(1,fnum):
            sum1[0], sum2[0] = 0, 0
            sum1[1], sum2[1] = 0, 0
            sum1[2], sum2[2] = 0, 0
                    
            for m in range(1):
                
                f1[0] = (m-n*s[0])*math.pi
                f2[0] = (m+n*s[0])*math.pi
                f1[1] = (m-n*s[1])*math.pi
                f2[1] = (m+n*s[1])*math.pi
                f1[2] = (m-n*s[2])*math.pi
                f2[2] = (m+n*s[2])*math.pi
                
                if abs(f1[0])<1e-20 or abs(f2[0])<1e-20:
                    sum1[0] += Am1[0,m]*( math.sin(2*n*math.pi*s[0])/(2*n*math.pi) + s[0] )
                else:
                    sum1[0] += Am1[0,m]*s[0]*( f1[0]*math.sin(f2[0]) + f2[0]*math.sin(f1[0]) )/(f1[0]*f2[0])
                    
                if abs(f1[1])<1e-20 or abs(f2[1])<1e-20:
                    sum1[1] += Am1[1,m]*( math.sin(2*n*math.pi*s[1])/(2*n*math.pi) + s[1] )
                else:
                    sum1[1] += Am1[1,m]*s[1]*( f1[1]*math.sin(f2[1]) + f2[1]*math.sin(f1[1]) )/(f1[1]*f2[1])
                    
                if abs(f1[2])<1e-20 or abs(f2[2])<1e-20:
                    sum1[2] += Am1[2,m]*( math.sin(2*n*math.pi*s[2])/(2*n*math.pi) + s[2] )
                else:
                    sum1[2] += Am1[2,m]*s[2]*( f1[2]*math.sin(f2[2]) + f2[2]*math.sin(f1[2]) )/(f1[2]*f2[2])
                    
                F1[0] = (m+n-n*s[0])*math.pi
                F2[0] = (m-n+n*s[0])*math.pi
                F1[1] = (m+n-n*s[1])*math.pi
                F2[1] = (m-n+n*s[1])*math.pi
                F1[2] = (m+n-n*s[2])*math.pi
                F2[2] = (m-n+n*s[2])*math.pi
                
                if abs(F1[0])<1e-20 or abs(F2[0])<1e-20:
                    sum2[0] += Am2[0,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[0]-1))/(n*math.pi) + math.cos(n*math.pi*s[0])*(1-s[0]) )
                else:
                    sum2[0] += Am2[0,m]*2*(-1)**n*(1-s[0])*( math.cos(F2[0]/2)*math.sin(F1[0]/2)/F1[0] + math.cos(F1[0]/2)*math.sin(F2[0]/2)/F2[0] )
                
                if abs(F1[1])<1e-20 or abs(F2[1])<1e-20:
                    sum2[1] += Am2[1,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[1]-1))/(n*math.pi) + math.cos(n*math.pi*s[1])*(1-s[1]) )
                else:
                    sum2[1] += Am2[1,m]*2*(-1)**n*(1-s[1])*( math.cos(F2[1]/2)*math.sin(F1[1]/2)/F1[1] + math.cos(F1[1]/2)*math.sin(F2[1]/2)/F2[1] )
                    
                if abs(F1[2])<1e-20 or abs(F2[2])<1e-20:
                    sum2[2] += Am2[2,m]*( (-1)**(n+1)*math.sin(n*math.pi*(s[2]-1))/(n*math.pi) + math.cos(n*math.pi*s[2])*(1-s[2]) )
                else:
                    sum2[2] += Am2[2,m]*2*(-1)**n*(1-s[2])*( math.cos(F2[2]/2)*math.sin(F1[2]/2)/F1[2] + math.cos(F1[2]/2)*math.sin(F2[2]/2)/F2[2] )
                
            Bm[0,n] = sum1[0] + sum2[0]
            Bm[1,n] = sum1[1] + sum2[1]
            Bm[2,n] = sum1[2] + sum2[2]
        
        # 2nd Layer    
        L2 = 51239.342*u
        w2 = 313.44465*u  
        a2 = w2*h                    
        for n in range(fnum):
            Bm[0,n] = Bm[0,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[6]/a2*w2/Di) )
            Bm[1,n] = Bm[1,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[7]/a2*w2/Di) )
            Bm[2,n] = Bm[2,n]*math.exp(-(n*math.pi)**2*L2/(w2*Q[8]/a2*w2/Di) )
        
        # Psi Mixer   
        s = cuda.local.array((1), dtype=float32)
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
        
        conc = cuda.local.array((101), dtype=float32)
        errc = 0
        for m in range(len(et)):
            for n in range(fnum):
                conc[m] += Cm[0,n]*math.cos(n*math.pi*et[m])                
            errc += abs(conc[m] - descon[m])
            
        fitness[x,y] = errc        
            
        

#%%

# GA Configurations
num_population = 64
num_individual = 128
num_gene = 12
num_generation = 1000
mutation_rate = 50/100

# Define Bounds
conc_lb = np.zeros((1,6))
conc_ub = np.ones((1,6))
volt_lb = np.zeros((1,6))
volt_ub = 1000*np.ones((1,6))
lower_limit = np.reshape(np.array([conc_lb, volt_lb]), (1, num_gene))
upper_limit = np.reshape(np.array([conc_ub, volt_ub]), (1, num_gene))

# Matrix Initialization
fitness = np.zeros((num_population, num_individual)).astype(np.float32)
offspring_sol = np.zeros((num_population, num_individual, num_gene)).astype(np.float32)
et = np.linspace(0, 1, 101).astype(np.float32)

# Initialize GPU Arrays
d_fit = cuda.device_array_like(fitness)
d_rankpop = cuda.device_array_like(offspring_sol)
d_offspring = cuda.device_array_like(offspring_sol)
d_descon = cuda.to_device(descon.astype(np.float32))
d_R = cuda.to_device(R)
d_et = cuda.to_device(et)
rng_states = create_xoroshiro128p_states(num_individual, seed=1)

# Define GPU Threads and Blocks
threadsperblock = (2, 16)
blockspergrid = (num_population//threadsperblock[0], num_individual//threadsperblock[1])


#%% Genetic Algorithm

# Creating the Initial Population
new_population = population(num_population, num_individual, num_gene, upper_limit, lower_limit)
# new_population[0,0,:6] = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
# new_population[0,0,6:] = np.array([500, 500, 500, 500, 500, 500])

# Transfer Population to GPU Device
d_pop = cuda.to_device(new_population)

start = timer()

for generation in range(num_generation):
    
    print("Generation : ", generation+1)
    
    # caculating the fitness of each chromosome in the population.    
    fitness_calculation[blockspergrid, threadsperblock](d_pop, d_fit, d_descon, d_R, d_et)
    cuda.synchronize()
    
    rank_population[blockspergrid, threadsperblock](d_pop, d_rankpop, d_fit)
    
    crossover_truncation[blockspergrid, threadsperblock](rng_states, d_rankpop, d_offspring, 0.25)
    # crossover_tournament[blockspergrid, threadsperblock](rng_states, d_rankedpop, d_offsprings, round(num_individuals*0.02))
    
    mrate = mutation_rate/(generation+1)
    if mrate < 0.01:
        mrate = 0.01
    
    # Adding some variations to the offsrping using mutation.
    mutation_random[blockspergrid, threadsperblock](rng_states, d_pop, d_rankpop, d_offspring, mrate)    
    
    if (generation%20 == 0):
        migration[blockspergrid, threadsperblock](d_pop, d_rankpop)
    
time_ga = timer() - start

print('Time taken for GA run is %f seconds.' % time_ga)
print('Time taken for a single generation is %f seconds.' % (time_ga/num_generation))


#%% Compute fitness for the last gen

fitness_calculation[blockspergrid, threadsperblock](d_pop, d_fit, d_descon, d_R, d_et)
cuda.synchronize()
    
# Copy the Result From Device
fitness = d_fit.copy_to_host()
# fitness[np.isnan(fitness)] = 9999
precon = d_descon.copy_to_host()

new_population = d_pop.copy_to_host()   

# Getting the best solution after iterating finishing all generations.
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.min(fitness))
best_pop = new_population[best_match_idx[0][0],best_match_idx[1][0],:]

print("Best solution : ", new_population[best_match_idx[0][0],best_match_idx[1][0],:])
print("Best solution fitness : ", fitness[best_match_idx[0][0],best_match_idx[1][0]])


#%% Plot Figures

V = np.zeros((10,))
V[0] = best_pop[6]
V[1] = best_pop[7]
V[2] = best_pop[8]
V[3] = best_pop[9]
V[4] = best_pop[10]
V[5] = best_pop[11]
Vout = 0

A1 = R2/R1 + R2/R1 + 1
A2 = R2/R1 + R2/R1 + 1
A3 = R2/R1 + R2/R1 + 1
A4 = R3/R2 + R3/R2 + R3/R2 + 1
D1 = V[0]*R2/R1 + V[1]*R2/R1
D2 = V[2]*R2/R1 + V[3]*R2/R1
D3 = V[4]*R2/R1 + V[5]*R2/R1

alpha = R3/R2*A2*A3 + R3/R2*A1*A3 + R3/R2*A1*A2 - A1*A2*A3*A4
beta = -R3/R2*D1*A2*A3 -R3/R2*D2*A1*A3 - R3/R2*D3*A1*A2
V16 = beta/alpha

V12 = (D1 + V16)/A1
V34 = (D2 + V16)/A2
V56 = (D3 + V16)/A3

Am1 = np.zeros(fnum)
Am2 = np.zeros(fnum)
Bm = np.zeros(fnum)

V[6] = V12
V[7] = V34
V[8] = V56
V[9] = V16

# Initial Concentrations
C = np.zeros((6,))
C[0] = best_pop[0]
C[1] = best_pop[1]
C[2] = best_pop[2]
C[3] = best_pop[3]
C[4] = best_pop[4]
C[5] = best_pop[5]

Dm = TripleTmixer(V, C, R)

et = np.linspace(0, 1, 101)
precon = np.zeros(len(et))

for m in range(len(et)):
    for n in range(fnum):
        precon[m] += Dm[n]*math.cos(n*math.pi*et[m])                 

# Plot desired concentration    
plt.plot(et, descon, 'b')
plt.plot(et, precon, '--r')
plt.xlabel('Normalized Width')
plt.ylabel('Normalized Concentration')
plt.legend(('Actual', 'Predicted'))
plt.show()


#%% Validate Fitness Vals

# fitval2 = np.zeros((num_individuals))
# for ndx in range(num_individuals):
    
#     C1 = new_population[ndx, 0]
#     C2 = new_population[ndx, 1]
#     V1 = new_population[ndx, 2]
#     V2 = new_population[ndx, 3]

#     Vout = 0
    
#     V12 = Pressure_Tnode(V1, V2, Vout, R1, R2, R3)
    
#     Am1 = np.zeros(fnum)
#     Am2 = np.zeros(fnum)
#     Bm = np.zeros(fnum)
    
#     Am1[0] = C1
#     Am2[0] = C2
    
#     # Am1 = Fluid_Channel(Am1, V1, V12, R1, L1, a1, w1)
#     # Am2 = Fluid_Channel(Am2, V2, V12, R2, L2, a2, w2)
#     Bm = Fluid_Tmixer(Am1, Am2, Bm, V1, V2, V12, R1, R2)
    
#     # Output Channel
#     Bm = Fluid_Channel(Bm, V12, Vout, R3, dloc, a3, w3)
    
#     et = np.linspace(0, 1, 101)
#     conc = np.zeros(len(et))
    
#     err = 0
#     for m in range(len(et)):
#         for n in range(len(Bm)):
#             conc[m] += Bm[n]*np.cos(n*math.pi*et[m])
#         err += abs(conc[m] - descon[m])
    
#     Cmat[ndx,:] = conc
#     fitval2[ndx] = err


# Emat = abs(Cmat - descon)
# fitval = np.sum(Emat, 1)




