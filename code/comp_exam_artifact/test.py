#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 8 03:35:43 2021

@author: Brian Kyanjo
"""
#-------------------------------------------------------------
#Test script 
#-------------------------------------------------------------
from numpy import *
from exact_solver import *
from approximate_solver import *

# Initial conditions
#The solver should supply an initialization routine to initialize  q(x,t)  at time  t=0 .
def h_init(x,hl,hr):    
    q0 = where(x < 0,hl,hr)
    return q0

def hu_init(x,hl,ul,hr,ur):    
    #q0 = zeros(x.shape)  
    q0 = where(x<0,hl*ul,hr*ur)
    return q0

def qinit(x,meqn,ql,qr):
    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)

    q = zeros((x.shape[0],meqn))
    q[:,0] = h_init(x,hl,hr)
    q[:,1] = hu_init(x,hl,ul,hr,ur)
    
    return q

# Boundary conditions
def bc_extrap(Q):
    """ Extend Q with extrapolation boundary conditions """
        
    Q_ext = concatenate((Q[[1,0],:], Q, Q[[-1,-2],:]))
    return Q_ext

# Problem test
def problem_test(case,itype):
    if itype == 0 and case > 6:
        print ('itype is not 1, invalid selection choose itype = 1 for dry state cases.')
    elif itype == 1 and case < 7:
        print ('itype is not 0, invalid selection choose itype = 0 for absence of dry state cases.')

    elif itype == 0: #no dry states 
        if case == 0:     #left going shock
            hl = 1
            hr = 1.5513875245483204
            ul = 0.5
            ur = 0

        elif(case == 1):  #right going shock
            hl = 1.5513875245483204
            hr = 1
            ul = 0.0
            ur = -0.5
        
        elif(case == 2):  #right going rarefaction
            hl = 0.5625
            hr = 1
            ul = 0
            ur = 0.5
        
        elif(case == 3):  #left going rarefaction
            hl = 2
            hr = 1.4571067811865475
            ul = 0
            ur = 0.41421356237309537
        
        elif(case == 4): #dam break
            hl = 2
            hr = 1
            ul = 0
            ur = 0
        
        elif (case == 5): #All rarefaction 
            hl = 1
            hr = 1
            ul = -0.5
            ur = 0.5
        
        elif (case == 6): #All shock 
            hl = 1
            hr = 1
            ul = 0.5
            ur = -0.5
        
    elif itype == 1: #presence of dry states 
        if case == 7: #left dry state
            hl = 0
            ul = 0
            hr = 1
            ur = 0
        elif case == 8: #middle dry state
            hl = .1
            ul = -.7
            hr = .1
            ur = 0.7
        elif case == 9: #right dry state
            hl = 1
            ul = 0
            hr = 0
            ur = 0
    ql = array([hl,hl*ul])
    qr = array([hr,hr*ur])
    return ql,qr
