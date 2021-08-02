#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:09:57 2021

@author: Brian KYANJO
"""
g =1

from numpy import *

def limiter(r,lim_choice='MC'):
    wlimiter = empty(r.shape)
    if lim_choice == None:
        wlimiter = ones(r.shape)
    elif lim_choice == 'minmod':        
        # wlimitr = dmax1(0.d0, dmin1(1.d0, r))
        wlimiter = maximum(0,minimum(1,r))
    elif lim_choice == 'superbee':
        # wlimitr = dmax1(0.d0, dmin1(1.d0, 2.d0*r), dmin1(2.d0, r))
        a1 = minimum(1,2*r)
        a2 = minimum(2,r)        
        wlimiter = maximum(0,maximum(a1,a2))
    if lim_choice == 'MC':
        # c = (1.d0 + r)/2.d0
        # wlimitr = dmax1(0.d0, dmin1(c, 2.d0, 2.d0*r))
        c = (1 + r)/2
        wlimiter = maximum(0,minimum(c,minimum(2,2*r)))
    elif lim_choice == 'vanleer':
        # wlimitr = (r + dabs(r)) / (1.d0 + dabs(r))
        wlimiter = (r + abs(r))/(1 + abs(r))
            
    return wlimiter

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
        
    Q_ext = concatenate((Q[1,0], Q, Q[-1,-2]))
    return Q_ext

def rp1_swe(qold1,qold2):
    """  Input : 
            Q_ext : Array of N+4 Q values.   Boundary conditions are included.
            
        Output : 
            waves  : Jump in Q at edges -3/2, -1/2, ..., N-1/2, N+1/2 (N+3 values total)
            speeds : Array of speeds (N+3 values)
            apdq   : Positive fluctuations (N+3 values)
            amdq   : Negative fluctuations (N+3 values)
        """    
        
    # This Riemann solver solves two-way wave equation.
    qold1_ext = bc_extrap(qold1) 
    qold2_ext = bc_extrap(qold2)
    
    # For most problems, the number of waves is equal to the number of equations
    mwaves = meqn

    d0 = qold1_ext[1:]- qold1_ext[:-1]
    d1 = qold2_ext[1:]- qold2_ext[:-1]
    
    mx = d0.shape[0]
    
    # Array of wave 1 and 2
    w1 = zeros(d0.shape)
    w2 = zeros(d0.shape)
    
    # Array of speed 1 and 2
    s1 = zeros((d0.shape[0],1))
    s2 = zeros((d0.shape[0],1))
    
    amdq = zeros(d0.shape)
    apdq = zeros(d0.shape)
    
    for i in range(1,mx):
        
        q1l = array([qold1_ext[i],qold2_ext[i]])
        q1r = array([qold1_ext[i+1],qold2_ext[i+1]])
           
        q2l = array([qold1_ext[i-1],qold2_ext[i-1]])
        q2r = array([qold1_ext[i],qold2_ext[i]])
            
        #at the intefaces
        hms1,ums1 = newton(q1l,q1r,g)
        hums1 = hms1*ums1
        q1e = array([hms1,hums1])
        
        hms2,ums2 = newton(q2l,q2r,g)
        hums2 = hms2*ums2
        q2e = array([hms2,hums2])

        #fluxe at the interface
        f1 = flux(q1e) #f_{i+1/2}    
        f2 = (flux(q2e)) #f_{i-1/2}

        
        q = array([qold1_ext[i],qold2_ext[i]]) #at edges
        
        #fluctuations
        amdq[i] = flux(q) - f2
        apdq[i] = f1 - flux(q)
        
        l2 = ums2 + sqrt(g*hms2) #2nd wave speed
        l1 = ums2 - sqrt(g*hms2) #1st wave speed
        
        # Eigenvalues
        #l1 = u_hat - c_hat        
        #l2 = u_hat + c_hat   
        
        # Eigenvectors
        r1 = array([1, l1])       
        r2 = array([1, l2])       
        
        R = array([r1,r2]).T
        
        # Vector of eigenvalues
        evals =  array([l1,l2])         

      
        #alpha
        alpha = linalg. inv(R)*(q2r-q2l)
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


def claw1(ax, bx, mx, Tfinal, nout, 
          meqn=1, \
          rp=None, \
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
    
    assert rp is not None,    'No user supplied Riemann solver'
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
        waves, speeds, amdq, apdq = rp(q_ext)
                
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