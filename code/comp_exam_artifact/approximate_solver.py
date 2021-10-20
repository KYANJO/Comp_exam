#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 07:18:43 2021

@author: Brian Kyanjo
"""

#-------------------------------------------------------------
#Approximate Riemann solver for Shallow water equations script
#-------------------------------------------------------------
'''
Parameters used:
----------------
ql  - Is an array containing height(hl) and momentum(hul) left states ([hl,hul])
qr  - Is an array containing height(hr) and momentum(hur) right states ([hr,hur])
qm  - Is an array containing height(hm) and momentum(hum) intermidiate states ([hm,hum])
qms - Is an array containing height(hms) and momentum(hums) intermidiate shock states ([hms,hums])
qmr - Is an array containing height(hmr) and momentum(humr) intermidiate rarefaction states ([hmr,humr])
g   - gravity
'''

g = 1

from numpy import *

#flux term
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


#Riemann Solvers:

#------------------------------------------------------------------
# Roe-solver: uses the Roe average to determine the waves and speeds
# see brian_kyanjo_synthesis_duplicate.pdf for the mathematics
# behind the method. 
#------------------------------------------------------------------
def rp0_swe(Q_ext,meqn):
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
    
    # For most problems, the number of waves is equal to the number of equations
    mwaves = meqn

    d0 = delta[:,[0]]
    d1 = delta[:,[1]]
    
    h = Q_ext[:,0]
    u = Q_ext[:,1]/Q_ext[:,0]
    n = delta.shape[0]
    
    # Array of wave 1 and 2
    w1 = zeros(delta.shape)
    w2 = zeros(delta.shape)
  
    # Array of speed 1 and 2
    s1 = zeros((delta.shape[0],1))
    s2 = zeros((delta.shape[0],1))
   
    for i in range(1,n):
        u_hat = (sqrt(h[i-1])*u[i-1]+sqrt(h[i])*u[i])/(sqrt(h[i-1])+sqrt(h[i]))
        h_bar = (1/2)*(h[i-1]+h[i])
        c_hat = sqrt(g*h_bar) 
        
        # Eigenvalues
        l1 = u_hat - c_hat        
        l2 = u_hat + c_hat   

        # Eigenvectors
        r1 = array([1, l1])       
        r2 = array([1, l2])          
        
        R = array([r1,r2]).T
        
        # Vector of eigenvalues
        evals =  array([l1,l2])     

        # Solve R*alpha = delta to get a1=alpha[0], a2=alpha[1]
        a1 = ((u_hat+c_hat)*d0[i-1]-d1[i-1])/(2*c_hat)
        a2 = (-(u_hat-c_hat)*d0[i-1]+d1[i-1])/(2*c_hat)
        
        # Wave and speed 1
        w1[i-1] = a1*R[:,[0]].T
        s1[i-1] = evals[0]

        # Wave and speed 2
        w2[i-1] = a2*R[:,[1]].T
        s2[i-1] = evals[1]
    
    waves = (w1,w2)             # P^th wave at each interface
    speeds = (s1,s2)            # Speeds at each interface

    # Fluctuations
    amdq = zeros(delta.shape)
    apdq = zeros(delta.shape)
    for mw in range(mwaves):
        sm = where(speeds[mw] < 0, speeds[mw], 0)
        amdq += sm*waves[mw]
        
        sp = where(speeds[mw] > 0, speeds[mw], 0)
        apdq += sp*waves[mw]
    
    return waves,speeds,amdq,apdq
#end of the Roe-solver

#-------------------------------------------------------------
# Riemann solver based on flux decompostion of waves. 
# It uses the exact solver to determine the middle state at each 
# interface
# see brian_kyanjo_synthesis_duplicate.pdf for the mathematics
# behind the method. 
#--------------------------------------------------------------
def rp1_swe(Q_ext,exact):
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
            
        #at each inteface
        hms = exact(ql,qr,0,0,g)
        hums = exact(ql,qr,0,1,g)
        ums = hums/hms
        
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
        w1[i] = a1*R[:,[0]].T
        s1[i] = evals[0]

        # Wave and speed 2
        w2[i] = a2*R[:,[1]].T
        s2[i] = evals[1]
    
    waves = (w1,w2)             # P^th wave at each interface
    speeds = (s1,s2)      
     
    return waves,speeds,amdq,apdq    
#End of the flux decompositon solver

#-------------------------------------------------------------
# Riemann solver based on f-wave approach with the source term: 
# see brian_kyanjo_synthesis_duplicate.pdf for the mathematics
# behind the method. 
#--------------------------------------------------------------
def rp2_swe(Q_ext,exact,x,dx):
    """  Input : 
            Q_ext : Array of N+4 Q values.   Boundary conditions are included.
            
        Output : 
            waves  : Jump in Q at edges -3/2, -1/2, ..., N-1/2, N+1/2 (N+3 values total)
            speeds : Array of speeds (N+3 values)
            apdq   : Positive fluctuations (N+3 values)
            amdq   : Negative fluctuations (N+3 values)
        """    
    # For most problems, the number of waves is equal to the number of equations
    mwaves = 2
    
     # jump in Q at each interface
    delta = Q_ext[1:,:]-Q_ext[:-1,:]

    d0 = delta[:,[0]]
    d1 = delta[:,[1]]
    
    h = Q_ext[:,0]
    u = Q_ext[:,1]/Q_ext[:,0]
    
    qold1 = Q_ext[:,0]
    qold2 = Q_ext[:,1]

    mx = delta.shape[0]
    
    # Array of wave 1 and 2
    z1 = zeros(delta.shape)
    z2 = zeros(delta.shape)
    
    # Array of speed 1 and 2
    s1 = zeros((delta.shape[0],1))
    s2 = zeros((delta.shape[0],1))
    
    #amdq = zeros(delta.shape)
    #apdq = zeros(delta.shape)
    
    for i in range(1,mx):
        
        #ql = array([qold1[i],qold2[i]])
        #qr = array([qold1[i+1],qold2[i+1]]) #at edges
            
         #at the intefaces
         #hms,ums = newton(hl,hr,ul,ur,g)
        #hms,ums = newton(ql,qr,g)
        #hms = exact(ql,qr,0,0,g)
        #hums = exact(ql,qr,0,1,g)
        #ums = hums/hms
        #hums = hms*ums
        
#         #state at the interface
        #qm = array([hms,hums])
        
#         #fluctuations
        #amdq[i] = flux(qm) - flux(ql)
        #apdq[i] = flux(qr) - flux(qm)
        
#         l1 = ums - sqrt(g*hms) #1st wave speed
#         l2 = ums + sqrt(g*hms) #2nd wave speed

        # use roe averages for middle state or just simply arithmetic averages e.g. hm = (hl + hr)/2
        #hm = (ql[0] + qr[0])/2
        #hum = (ql[1] + qr[1])/2
        #um = hum/hm
        
        #l1 = um - sqrt(g*hm) #1st wave speed
        #l2 = um + sqrt(g*hm) #2nd wave speed
        u_hat = (sqrt(h[i-1])*u[i-1]+sqrt(h[i])*u[i])/(sqrt(h[i-1])+sqrt(h[i]))
        h_bar = (1/2)*(h[i-1]+h[i])
        c_hat = sqrt(g*h_bar) 
        
        # Eigenvalues
        l1 = u_hat - c_hat        
        l2 = u_hat + c_hat 

        # Compute R and speeds based on these averages.

        # Eigenvectors
        r1 = array([1, l1])       
        r2 = array([1, l2])  
        
        # Matrix of eigenvalues
        R = array([r1,r2]).T
        
        # Vector of eigenvalues
        evals =  array([l1,l2])        
        
        ql = array([h[i-1],h[i-1]*u[i-1]])
        qr = array([h[i],h[i]*u[i]])

        #flux-based wave decomposition
        #beta_i+1/2
        beta = linalg. inv(R)*(flux(qr)-flux(ql) - dx*psi(x,g,h_bar,dBdx))
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
   
    # Fluctuations
    amdq = zeros(delta.shape)
    apdq = zeros(delta.shape)
    for m in range(mwaves):
        for mw in range(mwaves):
            sm = where(speeds[mw] < 0, speeds[mw], 0)
            sp = where(speeds[mw] > 0, speeds[mw], 0)
            #if speeds[mw].all() < 0:
            amdq += sm*fwaves[mw]
            #elif speeds[mw].all() > 0:
            apdq += sp*fwaves[mw]
            #else:
              #  amdq += fwaves[mw]*0.5
               # apdq += fwaves[mw]*0.5
            #amdq += (speeds[mw]==0)*fwaves[mw]*0.5
            #apdq += (speeds[mw]==0)*fwaves[mw]*0.5
            #sp = where(speeds[mw] > 0, speeds[mw], 0)
            #apdq += sign(sp)*fwaves[mw]
                
     
    return fwaves,speeds,amdq,apdq
#End of Riemans solvers


# ----------------------------------------------------
# Wave propagation algorithm
# 
# Adapted from the Clawpack package (www.clawpack.org)
# -----------------------------------------------------

def claw(ax, bx, mx, Tfinal, nout,ql,qr, 
          meqn=1, \
          exact=None,\
          solver=None,\
          qinit=None, \
          bc=None, \
          limiter_choice='MC',    
          second_order=True):

    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/(hl)
    ur = hur/(hr)
    
    dx = (bx-ax)/mx
    xe = linspace(ax,bx,mx+1)  # Edge locations
    xc = xe[:-1] + dx/2       # Cell-center locations

    # For many problems we can assume mwaves=meqn.
    mwaves = meqn
        
    # Temporal mesh
    t0 = 0
    tvec = linspace(t0,Tfinal,nout+1)
    dt = Tfinal/nout
    
    assert solver is not None,    'No user supplied Riemann solver'
    assert qinit is not None, 'No user supplied initialization routine'
    assert bc is not None,    'No user supplied boundary conditions'

    # Initial the solution
    q0 = qinit(xc,meqn,ql,qr)    # Should be [size(xc), meqn]
    
    # Store time stolutions
    Q = empty((mx,meqn,nout+1))    # Doesn't include ghost cells
    Q[:,:,0] = q0

    q = q0
    
    dtdx = dt/dx
    t = t0
    
    for n in range(0,nout):
        t = tvec[n]

        # solver: 0 - Roe solver
        # solver: 1 - flux formulation solver
        # solver: 2 - f-wave approach(with source term) solver
        if solver == 0:
            # Add 2 ghost cells at each end of the domain;  
            q_ext = bc(q)
            # Get waves, speeds and fluctuations from the solver 
            waves, speeds, amdq, apdq = rp0_swe(q_ext,meqn)

            # First order update
            q = q - dtdx*(apdq[1:-2,:] + amdq[2:-1,:])
            
            # Second order corrections (with possible limiter)
            if second_order:    
                cxx = zeros((q.shape[0]+1,meqn))  # Second order corrections defined at interfaces
                for p in range(mwaves):
                    sp = speeds[p][1:-1]   # Remove unneeded ghost cell values added by Riemann solver.
                    wavep = waves[p]
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
                        
                    cxx += 0.5*abs(sp)*(1 - abs(sp)*dtdx)*wavep[1:-1,:]*wlimiter
                    
                Fp = cxx  # Second order derivatives
                Fm = cxx
            
                # update with second order corrections
                q -= dtdx*(Fp[1:,:] - Fm[:-1,:])
    
        elif solver == 1:

             # Add 2 ghost cells at each end of the domain;  
            q_ext = bc(q)

            # Get waves, speeds and fluctuations
            waves, speeds, amdq, apdq = rp1_swe(q_ext,exact)
                    
            # First order update
            q = q - dtdx*(apdq[1:-2,:] + amdq[2:-1,:])
            
            # Second order corrections (with possible limiter)
            if second_order:    
                cxx = zeros((q.shape[0]+1,meqn))  # Second order corrections defined at interfaces
                for p in range(mwaves):
                    sp = speeds[p][1:-1]   # Remove unneeded ghost cell values added by Riemann solver.
                    wavep = waves[p]
                    
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
                        
                    cxx += 0.5*abs(sp)*(1 - abs(sp)*dtdx)*wavep[1:-1,:]*wlimiter
                    
                    #cxx += 0.5*abs(sp)*(1 - abs(sp)*dtdx)*wavep[1:-1,:]
                    
                Fp = cxx  # Second order derivatives
                Fm = cxx
            
                # update with second order corrections
                q -= dtdx*(Fp[1:,:] - Fm[:-1,:])

        elif solver == 2:
 
            # Add 2 ghost cells at each end of the domain;  
            q_ext = bc(q)

            # Get waves, speeds and fluctuations
            fwaves, speeds, amdq, apdq = rp2_swe(q_ext,exact,xc,dx)
                    
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

    if solver == 0:
        print('solver used is: Roe-solver')
        print("dt = {:.4e}".format(dt))
        print("Number of time steps = {}".format(nout))
            
    elif solver == 1:
        print('solver used is: flux-based')
        print("dt = {:.4e}".format(dt))
        print("Number of time steps = {}".format(nout))
    elif solver == 2:
        print('solver used is: f-wave')
        print("dt = {:.4e}".format(dt))
        print("Number of time steps = {}".format(nout))

    return Q, xc, tvec



