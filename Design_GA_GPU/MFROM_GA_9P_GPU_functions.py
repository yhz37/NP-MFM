#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:44:41 2021

@author: seonghyeonhong
"""

import math
import numpy as np
from numba import cuda, float32, jit
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

@jit
def resistance_TripleTMixer():
    
    # Micrometer
    u = 1e-6
    # Channel Depth
    h = 60*u
    # Viscosity
    mu = 0.001
    # Order of Fourier Coefficient
    fnum = 100
    
    # Channel Geometry    
    # Channel Layer 1
    L1 = 4000*u
    w1 = 110.82*u
    b1 = w1/h    
    # Channel Layer 2
    L2 = 51239.342*u
    w2 = 313.44465*u
    b2 = w2/h    
    # Channel Layer 3
    L3 = 10000*u
    w3 = 1200*u
    b3 = w3/h    
    
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
    
    return R.astype(np.float32)


@jit
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
            

# Truncation Selection
@cuda.jit
def crossover_truncation(rng_states, rankpop, offspring, tratio):

    x, y = cuda.grid(2)    
    T = round(tratio*rankpop.shape[2])
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
    
    # only mutate 25% of non-elite    
    # mutate ration term
    muprob = xoroshiro128p_uniform_float32(rng_states, y)
    if muprob <= 0.25 and y >= round(pop.shape[1]*0.02): 
        x1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(2))
        pop[x,y,x1] += xoroshiro128p_normal_float32(rng_states, y)*mrate*0.3
        
        # Lower and Upper Bounds
    for i in range(3):
        if pop[x,y,i] < 0.7:
            pop[x,y,i] = 0.7
        elif pop[x,y,i] > 1:
            pop[x,y,i] = 1
                             
    muprob = xoroshiro128p_uniform_float32(rng_states, y)
    if muprob <= 0.25 and y >= round(pop.shape[1]*0.02): 
    
        x1 = round(xoroshiro128p_uniform_float32(rng_states, y)*(2)) + 3
        x2 = round(xoroshiro128p_uniform_float32(rng_states, y)*(2)) + 6
        pop[x,y,x1] += xoroshiro128p_normal_float32(rng_states, y)*mrate*1
        pop[x,y,x2] += xoroshiro128p_normal_float32(rng_states, y)*mrate*1
        
        # Lower and Upper Bounds
        for i in range(3,pop.shape[2]):
            if pop[x,y,i] < 0:
                pop[x,y,i] = 0
            elif pop[x,y,i] > 1:
                pop[x,y,i] = 1


@cuda.jit    
def fitness_TripleTMixer(pop, fitness, alp_des, R, mZ, mS, mSsc, mYsc, mgamma, mbeta, mtheta):
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
    dloc = 2000*u
    w3 = 1200*u
    a3 = w3*h                    
    for n in range(fnum):
        Cm[0,n] = Cm[1,n]*math.exp(-(n*math.pi)**2*dloc/(w3*Q[9]/a3*w3/Di) )      
    
    # Reduced order multi-fidelity surrogate model    
    alp_pbcm = cuda.local.array((21), dtype=float32)    
    alp_cfd = cuda.local.array((25), dtype=float32)
    fval = cuda.local.array((22), dtype=float32)
    dx = cuda.local.array((1500,21), dtype=float32)
    rval = cuda.local.array((1500), dtype=float32)
    
    for m in range(21):
        for n in range(fnum):
            alp_pbcm[m] += mZ[m,n]*Cm[0,n]
    
    # Normalization and define fval
    fval[0] = 1
    for m in range(21):
        alp_pbcm[m] = (alp_pbcm[m]-mSsc[0,m])/mSsc[1,m]
        fval[m+1] = alp_pbcm[m]
        
    for m in range(21):
        for n in range(1500):
            dx[n,m] = (alp_pbcm[m] - mS[n,m])**2
    
    for m in range(1500):
        for n in range(21):
            rval[m] += dx[m,n]*(-mtheta[0,n])
            # rval[m] += 0.01*(-mtheta[0,n])
        rval[m] = math.exp(rval[m])
    
    for m in range(25):
        for n in range(22):
            alp_cfd[m] += fval[n]*mbeta[n,m]
        for n in range(1500):
            alp_cfd[m] += mgamma[m,n]*rval[n]
    
    errc = 0
    for m in range(25):
        alp_cfd[m] = mYsc[0,m] + alp_cfd[m]*mYsc[1,m]    
        errc += (alp_des[m] - alp_cfd[m])**2        
            
    # fitness[x,y] = 1
    fitness[x,y] = math.sqrt(errc)



        
        

@jit    
def TripleTMixer(pop, R, et):
               
    Di = 1e-10
    u = 1e-6    
    h = 60*u        
    Vout = 0
    fnum = 151
    
    C = np.zeros((6,)).astype(np.float32)
    V = np.zeros((10,)).astype(np.float32)
    
    C[0] = pop[0]
    C[1] = pop[1]
    C[2] = pop[2]
    C[3] = pop[3]
    C[4] = pop[4]
    C[5] = pop[5]
    V[0] = pop[6]
    V[1] = pop[7]
    V[2] = pop[8]
    V[3] = pop[9]
    V[4] = pop[10]
    V[5] = pop[11]    
    
    # Compute Node Pressures
    A1 = R[1]/R[0] + R[1]/R[0] + 1
    A2 = R[1]/R[0] + R[1]/R[0] + 1
    A3 = R[1]/R[0] + R[1]/R[0] + 1
    A4 = R[2]/R[1] + R[2]/R[1] + R[2]/R[1] + 1
    D1 = V[0]*R[1]/R[0] + V[1]*R[1]/R[0]
    D2 = V[2]*R[1]/R[0] + V[3]*R[1]/R[0]
    D3 = V[4]*R[1]/R[0] + V[5]*R[1]/R[0]
    
    aa = R[2]/R[1]*(A2*A3 + A1*A3 + A1*A2) - A1*A2*A3*A4
    bb = -R[2]/R[1]*(D1*A2*A3 + D2*A1*A3 + D3*A1*A2)
    V[9] = bb/aa    
    V[6] = (D1 + V[9])/A1
    V[7] = (D2 + V[9])/A2
    V[8] = (D3 + V[9])/A3
    
    Q = np.zeros((10,)).astype(np.float32)    
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
    s = np.zeros((3,)).astype(np.float32)    
    s[0] = Q[0]/(Q[0]+Q[1])
    s[1] = Q[2]/(Q[2]+Q[3])
    s[2] = Q[4]/(Q[4]+Q[5])
            
    Am1 = np.zeros((3, 1)).astype(np.float32)
    Am2 = np.zeros((3, 1)).astype(np.float32)
    Bm = np.zeros((3, 151)).astype(np.float32)
    Cm = np.zeros((2, 151)).astype(np.float32)    
        
    Am1[0,0] = C[0]
    Am2[0,0] = C[1]
    Am1[1,0] = C[2]
    Am2[1,0] = C[3]
    Am1[2,0] = C[4]
    Am2[2,0] = C[5]
    
    Bm[0,0] = Am1[0,0]*s[0] + Am2[0,0]*(1-s[0])
    Bm[1,0] = Am1[1,0]*s[1] + Am2[1,0]*(1-s[1])
    Bm[2,0] = Am1[2,0]*s[2] + Am2[2,0]*(1-s[2]) 
   
    sum1 = np.zeros((3,)).astype(np.float32)
    sum2 = np.zeros((3,)).astype(np.float32)
    f1 = np.zeros((3,)).astype(np.float32)
    f2 = np.zeros((3,)).astype(np.float32)
    F1 = np.zeros((3,)).astype(np.float32)
    F2 = np.zeros((3,)).astype(np.float32)    
    
    # Triple T Mixers
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
    s = 0
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
    dloc = 2000*u
    w3 = 1200*u
    a3 = w3*h                    
    for n in range(fnum):
        Cm[0,n] = Cm[1,n]*math.exp(-(n*math.pi)**2*dloc/(w3*Q[9]/a3*w3/Di) )
    
    
    # Surrogate Model
    # alp_pbcm
    
    
    conc = np.zeros((len(et),)).astype(np.float32)        
    for m in range(len(et)):
        for n in range(fnum):
            conc[m] += Cm[0,n]*math.cos(n*math.pi*et[m])                
    
    return conc



