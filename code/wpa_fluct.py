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

def wpa_f(ax,bx,mx,meqn,Tfinal,nout,g,\
            newton=None,\
            qinit=None):
    
    #spartial mesh
    dx = (bx-ax)/mx
    xe = linspace(ax,bx,mx+1)  #Edge locations
    xc = xe[:-1] + dx/2        #Cell-center locations
    
    #Temporal mesh
    t0 = 0
    tvec = linspace(t0,Tfinal,nout)
    dt = Tfinal/nout
    
    dtdx = dt/dx
    #assert rp is not None,    'No user supplied Riemann solver'
    assert qinit is not None, 'No user supplied initialization routine'
   
    qnew1 = zeros(mx)
    qnew2 = zeros(mx)
    
    #Intial solution
    q0 = qinit(xc,meqn)
    
   # Store time stolutions
    Q = empty((mx,meqn,nout+1))    # Doesn't include ghost cells
    
    Q[:,:,0] = q0
    
    qold1 = q0[:,0]
    qold2 = q0[:,1]
    
    for n in range(nout):
        
        for i in range(mx):
            if i == mx-1:
                q1l = array([qold1[i],qold2[i]])
                q1r = array([qold1[i],qold2[i]])
            else:
                q1l = array([qold1[i],qold2[i]])
                q1r = array([qold1[i+1],qold2[i+1]])
            
            if i == 0:
                q2l = array([qold1[i],qold2[i]])
                q2r = array([qold1[i],qold2[i]])
            else:
                q2l = array([qold1[i-1],qold2[i-1]])
                q2r = array([qold1[i],qold2[i]])
            
            #at the inteface
            hms1,ums1 = newton(q1l,q1r,g)
            hums1 = hms1*ums1
            q1e = array([hms1,hums1])
    
            f1 = flux(q1e) #f_{i+1/2}    
    
            hms2,ums2 = newton(q2l,q2r,g)
            hums2 = hms2*ums2
            q2e = array([hms2,hums2])
    
            f2 = (flux(q2e)) #f_{i-1/2}

            
            q = array([qold1[i],qold2[i]]) #at edges
            
            #fluctuations
            amdq = flux(q) - f2
            apdq = f1 - flux(q)
            
           #soln at N+1
            qnew1[i] = qold1[i] - (dt/dx)*(apdq[0] + amdq[0])
    
            qnew2[i] = qold2[i] - (dt/dx)*(apdq[1] + amdq[1])

    
        Q[:,0,n+1] = qnew1
        Q[:,1,n+1] = qnew2
        
        #overwrite the soln
        qold1 = qnew1.copy()
        qold2 = qnew2.copy()
  
    return Q,xc,tvec
    