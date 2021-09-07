#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 07:18:43 2021

@author: Brian Kyanjo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:09:57 2021

@author: Brian KYANJO
"""
g = 1

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

#limiters
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

#derivative of the bathemetry
def dBdx(x):
    if abs(x-0.5) < 0.1:
        return (0.25*pi/0.1)*(sin((pi*(x - 0.5)/0.1) + 1))
    else:
        return 0

#source term
def psi(x,g,hm,dBdx):
    for i in x:
        a  = -g*hm*dBdx(i)
        return array([0,a])
    
#--------------------------------------------------------------------
#Godunov solver and LxF solver
#
#
#--------------------------------------------------------------------

# Global data needed for Riemann solver and initialization routine

def rp2_swe(Q_ext,newton,x,dx):
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

    d0 = delta[:,[0]]
    d1 = delta[:,[1]]
    
    qold1 = Q_ext[:,0]
    qold2 = Q_ext[:,1]
    
    mx = delta.shape[0]
    
    # Array of wave 1 and 2
    z1 = zeros(delta.shape)
    z2 = zeros(delta.shape)
    
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
        
        #flux-based wave decomposition
        #beta_i+1/2
        beta = linalg. inv(R)*(flux(qr)-flux(ql) - dx*psi(x,g,hms,dBdx))
        b1 = beta[0]
        b2 = beta[1]
        
        # f-wave and speed 1
        z1[i] = b1*R[:,[0]].T
        s1[i] = evals[0]

        # f-wave and speed 2
        z2[i] = b2*R[:,[1]].T
        s2[i] = evals[1]
    
    fwaves = (z1,z2)             # P^th wave at each interface
    speeds = (s1,s2)      
     
    return fwaves,speeds,amdq,apdq


def claw2(ax, bx, mx, Tfinal, nout, 
          meqn=1, \
          newton=None,\
          qinit=None, \
          bc=None, \
          limiter_choice='MC',    
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
        fwaves, speeds, amdq, apdq = rp2_swe(q_ext,newton,xc,dx)
                
        # First order update
        q = q - dtdx*(apdq[1:-2,:] + amdq[2:-1,:])
        
        # Second order corrections (with possible limiter)
        if second_order:    
            cxx = zeros((q.shape[0]+1,meqn))  # Second order corrections defined at interfaces
            for p in range(mwaves):
                sp = speeds[p][1:-1]   # Remove unneeded ghost cell values added by Riemann solver.
                wavep = fwaves[p]
                
                if limiter_choice is not None:
                    wl = sum(wavep[:-2,:] *wavep[1:-1,:],axis=1)  # Sum along dim=1
                    wr = sum(wavep[2:,:]  *wavep[1:-1,:],axis=1)
                    w2 = sum(wavep[1:-1,:]*wavep[1:-1,:],axis=1)
                
                    # Create mask to avoid dividing by 0. 
                    m = w2 > 0
                    r = ones(w2.shape)
                    
                    r[m] = where(sp[m,0] > 0,  wl[m]/w2[m],wr[m]/w2[m])

                    wlimiter = limiter(r,lim_choice=limiter_choice)
                    wlimiter.shape = sp.shape
                    
                else:
                    wlimiter = 1
                    
                cxx += 0.5*sign(sp)*(1 - abs(sp)*dtdx)*wavep[1:-1,:]*wlimiter
                
                #cxx += 0.5*abs(sp)*(1 - abs(sp)*dtdx)*wavep[1:-1,:]
                
            Fp = cxx  # Second order derivatives
            Fm = cxx
        
            # update with second order corrections
            q -= dtdx*(Fp[1:,:] - Fm[:-1,:])
        
        Q[:,:,n+1] = q
        
    return Q, xc, tvec



