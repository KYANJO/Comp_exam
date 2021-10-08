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

#--------------------------------------------------------------------
#Godunov solver and LxF solver
#
#
#--------------------------------------------------------------------

def wpa_f(ax,bx,mx,mq,meqn,Tfinal,nout,g,\
            newton=None,\
            second_order = True,\
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
    
    #for many problems we can assume mwaves=meqn
    mwaves = meqn
    
    #assert rp is not None,    'No user supplied Riemann solver'
    assert qinit is not None, 'No user supplied initialization routine'
    assert newton is not None, 'No user supplied newton solver'
    
   
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

            
            q = array([qold1[i],qold2[i]]) #at edges
            
            #fluctuations
            amdq = flux(q) - f2
            apdq = f1 - flux(q)
            
            #soln at N+1
            qnew1[i] = qold1[i] - (dtdx)*(apdq[0] + amdq[0])
    
            qnew2[i] = qold2[i] - (dtdx)*(apdq[1] + amdq[1])
            
            #second order corrections (without limiters)
            if second_order:
                #evaluate speeds and waves at the middle state
                #s_{i-1/2}
                l2 = ums2 + sqrt(g*hms2) #2nd wave speed
                l1 = ums2 - sqrt(g*hms2) #1st wave speed
                
                #s_{i+1/2}
                l22 = ums1 + sqrt(g*hms1) #2nd wave speed
                l11 = ums1 - sqrt(g*hms1) #1st wave speed
                
                s1 = array([l1,l2]) #s_{i-1/2}
                s2 = array([l11,l22]) #s_{i+1/2}
                
                #eigen vectors(waves) at {i-1/2}
                r1 = array([1,l1]) 
                r2 = array([1,l2]) 
                
                #eigen vectors(waves) at {i+1/2}
                r11 = array([1,l11]) 
                r22 = array([1,l22]) 
                
                R1 = array([r1,r2]).T
                R2 = array([r11,r22]).T
                
                #alpha
                a1 = linalg. inv(R1)*(q2r-q2l)
                a2 = linalg. inv(R2)*(q1r-q1l)
                
                #1st and 2nd waves w^{p}_{1-1/2} = alpha^{p}_{i-1/2}*r^{p}
                w1 = a1*R1[:,[0]].T
                w2 = a2*R2[:,[1]].T
                
                #second order corrections defined at interface
                Fp = 0 
                Fm = 0
                
                for p in range(mwaves):
                
                    #second order corrections
                    Fm += 0.5*abs(s1[p])*(1-abs(s1[p])*dtdx)*w1[p]
                    Fp += 0.5*abs(s2[p])*(1-abs(s2[p])*dtdx)*w2[p]
                
                #soln at N+1 
                qnew1[i] -= dtdx*(Fp[0] - Fm[0]) 
    
                qnew2[i] -= dtdx*(Fp[1] - Fm[1])                                     
    
        qq = array([qnew1,qnew2])
        Q[:,mq,n+1] = qq[mq]
        
        
        #overwrite the soln
        qold1 = qnew1.copy()
        qold2 = qnew2.copy()
  
    return Q,xc,tvec
    