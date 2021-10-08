#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:09:57 2021

@author: Brian KYANJO
"""
g =1

from numpy import *

#flux
def flux(q):
    '''
    input:
    -----
    q - state at the interface
    return:
    -------
    f - flux at the interface
    '''
    q1 = q[0]
    q2 = q[1]
    f = zeros(2)
    f[0] = q2
    f[1] = (((q2)**2)/q1) + (0.5*g*(q1)**2)
    return f

#--------------------------------------------------------------------
#Godunov solver and LxF solver
#
#
#--------------------------------------------------------------------

# Global data needed for Riemann solver and initialization routine


def bc_extrap(Q):
    """ Extend Q with extrapolation boundary conditions """
        
    Q_ext = concatenate((Q[[0,1],:], Q, Q[[-1,-2],:]))
    return Q_ext

def rp1_sw(Q_ext,newton):
    """  Input : 
            Q_ext : Array of N+4 Q values.   Boundary conditions are included.
            
        Output : 
            waves  : Jump in Q at edges -3/2, -1/2, ..., N-1/2, N+1/2 (N+3 values total)
            speeds : Array of speeds (N+3 values)
            apdq   : Positive fluctuations (N+3 values)
            amdq   : Negative fluctuations (N+3 values)
        """    
        
     # jump in Q at each interface
    delta = Q_ext[1:,:]-Q_ext[:-1,:]

    # -----------------------------------------------------------------------------------------
    # TODO : loop over each interface;  compute eigenvalues and eigenvectors at each interface. 
    # These will depend on Q_{i-1} and Q_{i}.  Use "Roe averages" to evaluate r1, r2, l1 and l2.  
    
    d0 = delta[:,[0]]
    d1 = delta[:,[1]]
    
    qold1 = Q_ext[:,0]
    qold2 = Q_ext[:,1]
    
    mx = delta.shape[0]
    
    # Array of wave 1 and 2
    w1 = zeros(delta.shape)
    w2 = zeros(delta.shape)
    
    # Array of speed 1 and 2
    s1 = zeros((delta.shape[0],1))
    s2 = zeros((delta.shape[0],1))
    
    amdq = zeros(delta.shape)
    apdq = zeros(delta.shape)
    
    for i in range(1,mx):
        
        ql = array([qold1[i],qold2[i]])
        qr = array([qold1[i+1],qold2[i+1]]) #at edges
            
        #at the intefaces
        hms,ums = newton(ql,qr,g)
        hums = hms*ums
        
        #state at the interface
        qm = array([hms,hums])
        
        #fluctuations
        amdq[i] = flux(qm) - flux(ql)
        apdq[i] = flux(qr) - flux(qm)
        
        l1 = ums - sqrt(g*hms) #1st wave speed
        l2 = ums + sqrt(g*hms) #2nd wave speed
        
        # Eigenvectors
        r1 = array([1, l1])       
        r2 = array([1, l2])  
        
        # Matrix of eigenvalues
        R = array([r1,r2]).T
        
        # Vector of eigenvalues
        evals =  array([l1,l2])        
        
        #alpha
        alpha = linalg. inv(R)*(qr-ql)
        a1 = alpha[0]
        a2 = alpha[1]
        
        # Wave and speed 1
        w1[i-1] = a1*R[:,[0]].T
        s1[i-1] = evals[0]

        # Wave and speed 2
        w2[i-1] = a2*R[:,[1]].T
        s2[i-1] = evals[1]
    
    waves = (w1,w2)             # P^th wave at each interface
    speeds = (s1,s2)      
     
    return waves,speeds,amdq,apdq


def claw(ax, bx, mx, Tfinal, nout, 
          meqn=1, \
          newton=None,\
          qinit=None, \
          bc=None, \
          second_order=True):

    dx = (bx-ax)/mx
    xe = linspace(ax,bx,mx+1)  # Edge locations
    xc = xe[:-1] + dx/2       # Cell-center locations

    # For many problems we can assume mwaves=meqn.
    mwaves = meqn
        
    # Temporal mesh
    t0 = 0
    tvec = linspace(t0,Tfinal,nout+1)
    dt = Tfinal/nout
    
   # assert rp is not None,    'No user supplied Riemann solver'
    assert qinit is not None, 'No user supplied initialization routine'
    assert bc is not None,    'No user supplied boundary conditions'

    # Initial the solution
    q0 = qinit(xc,meqn)    # Should be [size(xc), meqn]
    
    # Store time stolutions
    Q = empty((mx,meqn,nout+1))    # Doesn't include ghost cells
    Q[:,:,0] = q0

    q = q0
    
    dtdx = dt/dx
    t = t0
    for n in range(0,nout):
        t = tvec[n]
        
        # Add 2 ghost cells at each end of the domain;  
        q_ext = bc(q)

        # Get waves, speeds and fluctuations
        waves, speeds, amdq, apdq = rp1_sw(q_ext,newton)
                
        # First order update
        q = q - dtdx*(apdq[1:-2,:] + amdq[2:-1,:])
        
        # Second order corrections (with possible limiter)
        if second_order:    
            cxx = zeros((q.shape[0]+1,meqn))  # Second order corrections defined at interfaces
            for p in range(mwaves):
                sp = speeds[p][1:-1]   # Remove unneeded ghost cell values added by Riemann solver.
                wavep = waves[p]
                
                cxx += 0.5*abs(sp)*(1 - abs(sp)*dtdx)*wavep[1:-1,:]
                
            Fp = cxx  # Second order derivatives
            Fm = cxx
        
            # update with second order corrections
            q -= dtdx*(Fp[1:,:] - Fm[:-1,:])
        
        Q[:,:,n+1] = q
        
    return Q, xc, tvec

