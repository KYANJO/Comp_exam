#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:08:44 2021

@author: Brian KYANJO
"""
#-------------------------------------------------------
#Exact Riemann solver for Shallow water equations script
# see brian_kyanjo_synthesis_duplicate.pdf for the mathematics
# behind the method. 
#-------------------------------------------------------
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

from numpy import *

def pospart(x):
    '''
    Returns a value greater than zero
    '''
    return max(1e-15,x)

#All shock solution
def ff(hm,um,g,hl,hr,ul,ur):
    '''
    Input:
    ------
    hm - Intemediate hieght field
    um - Intemediate velocity field
    Returns:
    ------
    f1 - A curve that corresponds to a 2-shock (connects qm to qr)
    f2 - A curve that corresponds to a 1-shock (connects ql to qm)
    '''
    #hugoniot_locus
    f1 = (um - (ur + (hm-hr)*sqrt((g/2)*(1/hm + 1/pospart(hr))))) 
    f2 = (um - (ul - (hm-hl)*sqrt((g/2)*(1/hm + 1/pospart(hl)))))
    return f1,f2
    
#Derivative of function ff
def dff(hm,um,g,hl,hr,ul,ur):
    '''
    Input:
    ------
    hm - Intemediate hieght field
    um - Intemediate velocity field
    Returns:
    --------    
    f1h - Derivative of f1 wrt h
    f1u - Derivative of f1 wrt u
    f2h - Derivative of f2 wrt h
    f2u - Derivative of f2 wrt u
    '''
    f1h = (sqrt(2*g*(hm + hr)/(hm*pospart(hr))))*(-2*hm*(hm + hr) + hr*(hm - hr)) \
          / (4*hm*(hm + hr))
   
    f2h = (sqrt(2*g*(hm + hl)/(hm*pospart(hl))))*(2*hm*(hm + hl) + hl*(hl - hm)) \
          / (4*hm*(hm + hl))
          
    f1u = 1
    f2u = 1
    
    return f1h,f1u,f2h,f2u

#Jacobian
def J(dff,hm,um,g,hl,hr,ul,ur):
    '''
    Input:
    ------
    hm - Intemediate hieght field
    um - Intemediate velocity field
    Returns: A 2x2 matrix that contains derivatives of function ff wrt h and u.
    -------
    '''
    f1h,f1u,f2h,f2u = dff(hm,um,g,hl,hr,ul,ur)
    
    return array([[f1h,f1u],[f2h,f2u]])

#Inverse of Jacobian
def Jinv(hm,um,hl,hr,ul,ur,g):
    '''
    Input:
    ------
    hm - Intemediate hieght field
    um - Intemediate velocity field
    Returns: The inverse of the Jacobian matrix
    -------
    '''
    return linalg.inv(J(dff,hm,um,g,hl,hr,ul,ur))

def f(hm,um,g,hl,hr,ul,ur):
    '''
    Input:
    ------
    hm - Intemediate hieght field
    um - Intemediate velocity field
    Returns: An array of all shock solutions (f1 - 2-shock and f2 - 1-shock)
    -------
    '''
    f1,f2 = ff(hm,um,g,hl,hr,ul,ur)
    return array([f1,f2])

#shock wave solution
def Newton(ql,qr,g):
    '''
    Description: Newton solver used to generate all shock Riemann solution
    -----------
    Returns:
    -------
    hm - Intermediate shock hieght field
    um - Intermediate shock velocity field
    '''
    #max _iterations
    max_iter = 20
    #tolerance
    epsilon  = 1e-16
    
    #intial conditions (IC)
    ho = 0.01
    uo = 0.01
    
    #left state intial height and momentum field 
    hl = ql[0]
    hul = ql[1]
    
    #right state intial height and momentum field
    hr = qr[0]
    hur = qr[1]
    
    #left and right intial states velocities
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)

    #intial value
    vo = array([ho,uo])

    #Newton solver
    for i in range(max_iter):

        v1 = vo - Jinv(ho,uo,hl,hr,ul,ur,g)@f(ho,uo,g,hl,hr,ul,ur)

        if linalg.norm(v1-vo) < epsilon:
            break
            
        else:
            vo = v1
            ho = v1[0]
            uo = v1[1]
    
    #Intermediate fields
    hm = v1[0]
    um =v1[1]
    
    return hm,um

#All Rarefaction Riemann solution
def rare(ql,qr,g):
    '''
    Description: All Rarefaction Riemann solution
    -----------
    Returns:
    -------
    hm - Intermediate rarefaction hieght field
    um - Intermediate rarefaction velocity field
    '''

    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    

    hm = (1/16*g)*(ul - ur + 2*(sqrt(g*hl) + sqrt(g*hr)))**2 
    um = ul + 2*(sqrt(g*hl) - sqrt(g*hm))  #integral curve
    return hm,um

#shock speed
#location of the shock
def sl(h,ql,g):
    '''
    Description: Location of the shock
    -----------
    Input: 
    ------
    h - Intemediated hieght
    ql - Left state
    Returns: Left shock speed
    -------    
    '''
    hl = ql[0] #initial left height
    hul = ql[1] #initial left momentum
    
    ul = hul/pospart(hl)
    
    return (ul - (1/pospart(hl))*sqrt((g/2)*(hl*h*(hl+h))))

def sr(h,qr,g):
    '''
    Description: Location of the shock
    -----------
    Input: 
    ------
    h - Intemediated hieght
    qr - Right state
    Returns: Right shock speed
    -------    
    '''
    
    hr = qr[0] #intial right hieght
    hur = qr[1] #intial right momentum
    
    #intial right velocity
    ur = hur/pospart(hr)
    
    return (ur + (1/pospart(hr))*sqrt((g/2)*(hr*h*(hr+h))))

#eigen values
def lam1(h,u,g):
    '''
    Input: 
    ------
    h - Intemediated hieght
    u - Intemediated velocity 
    Returns: First family eigen value (wave speed)
    -------    
    '''
    return (u - sqrt(g*h))

def lam2(h,u,g):
    '''
     Input: 
    ------
    h - Intemediated hieght
    u - Intemediated velocity 
    Returns: second family eigen value (wave speed)
    -------    
    '''
    return (u + sqrt(g*h))

#Dry state velocity
def dry_velocity(ql,qr,g):
    '''
    Description:
    -----------
    These velocities are obtained when we encounter a zero intial state(dry state)
    either on the left or right region of the Riemann problem.
    Returns:
    -------
    d_vl - left state dry velocity
    d_vr - right state dry velocity
    '''
    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
    d_vl = ul + 2*sqrt(g*hl)
    d_vr = ur - 2*sqrt(g*hr)
    
    return d_vl,d_vr

#-------------------------------------------------------------
# Exact Riemann solver that returns an array of values for the
# entire simulation. 
# see brian_kyanjo_synthesis_duplicate.pdf for the mathematics
# behind the method. 
#-------------------------------------------------------------

def qexact(x,t,mq,ql,qr,g):
    '''
    Description:
    -----------
    Uses the Initial Riemann problem to find an intemediate state (qm) which either 
    the intial left or right state connects to it via any combination of shocks and
    rarefactions in the two families.
    
    For Riemann problems with an initial dry state on one side, the exact Riemann 
    solution contains only a single rarefaction connecting the wet to dry state.

    The evolving wet dry interface is therefore simply one edge of the rarefaction. 
    The propagation speed of this interface can be exactly determined using the 
    Riemann invariants of the corresponding characteristics field.
    
    Input:
    -----
    x  - array of spacial points 
    t  - array of temporal points
    mq - specifies the output (0 and 1 corresponds to h and hu respectively) 
    ql - left initial state
    qr - right initial state
    Returns:
    h  - array of hieght field values
    hu - array of momentum field values

    '''
    
    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
    
    #dry velocity
    d_vl,d_vr = dry_velocity(ql, qr,g)
    
    if t == 0:
        q1 = where(x<0,hl,hr)
        q2 = where(x<0,hl*ul,hr*ur)
        q3 = where(x<0,ul,ur)
    else:
        
        q1 = zeros(x.shape)
        q2 = zeros(x.shape)
        q3 = zeros(x.shape)      

        #dry middle state
        if d_vl < d_vr:
            hmr = 0
            umr = 0.5*(d_vl + d_vr) #wet-dry interface speed (arbitrary)
            humr = hmr*umr
            
            for i in range(len(x)):
                if x[i]<=lam1(hl,ul,g)*t:
                    h = hl
                    u = ul
                    hu = h*u
                elif lam1(hl,ul,g)*t <= x[i] and x[i]<=lam1(hmr,umr,g)*t: 
                    #inside the rarefaction
                    A = ul + 2*sqrt(g*hl)
                    h = (1/(9*g))*(A-(x[i]/t))**2
                    u = (x[i]/t) + sqrt(g*h)
                    hu = h*u
                       
                elif lam1(hmr,umr,g)*t<=x[i] and x[i]<=t*lam2(hmr, umr,g):
                    h = hmr
                    hu = hmr*umr
                    
                elif x[i]>=t*lam2(hmr, umr,g) and x[i]<=t*lam2(hr, ur,g):
                    #inside the rarefaction
                    A = ur - 2*sqrt(g*hr)
                    h = (1/(9*g))*(A-(x[i]/t))**2                    
                    u = (x[i]/t) - sqrt(g*h)
                    hu = h*u
                    
                else:
                    h = hr
                    hu = hr*ur
                    
                q1[i] = h
                q2[i] = hu
                q3[i] = u
                
        #dry left state (2-rarefaction only)
        elif hl == 0:
            hmr = 0
            umr = d_vr              #wet-dry interface speed
            humr = hmr*umr
        
            for i in range(len(x)):
                if x[i] <= t*lam2(hmr, umr,g) :
                    h = hl
                    u = ul
                    hu = h*u   
                    
                elif x[i]>=t*lam2(hmr, umr,g) and x[i]<=t*lam2(hr, ur,g):
                    #inside rarefaction
                    A = ur - 2*sqrt(g*hr)
                    h = (1/(9*g))*(A-(x[i]/t))**2
                    u = (x[i]/t) - sqrt(g*h)
                    hu = h*u
                    
                else:
                    h = hr
                    u = ur
                    hu = h*u
                
                q1[i] = h
                q2[i] = hu   
                q3[i] = u
            
                
        #dry right state (1-rarefaction only)
        elif hr == 0:
            hmr = 0
            umr = d_vl              #wet-dry interface speed
            humr = hmr*umr
           
            for i in range(len(x)):
                if x[i]<=lam1(hl,ul,g)*t:
                    h = hl
                    u = ul
                    hu = h*u
                    
                elif lam1(hl,ul,g)*t <= x[i] and x[i]<=lam1(hmr,umr,g)*t:
                    #inside the rarefaction
                    A = ul + 2*sqrt(g*hl)
                    h = (1/(9*g))*(A-(x[i]/t))**2                   
                    u = (x[i]/t) + sqrt(g*h)
                    hu = h*u
                    
                else:
                    h = hr
                    u = ur
                    hu = h*u
                    
                q1[i] = h
                q2[i] = hu
                q3[i] = u
                
        else:
        
            #for shock soln
            hms,ums = Newton(ql,qr,g)
            hums = hms*ums
            qms = array([hms,hums])
            
            #for rarefaction soln
            hmr,umr = rare(ql,qr,g)
            humr = hmr*umr
            qmr = array([hmr,humr])  
            
            for i in range(len(x)):
                if x[i] <= ums*t:
                    if hms>hl:
                        if x[i]<= sl(hms,ql,g)*t:
                            h = hl
                            u = ul
                            hu =h*u
                        else:
                            h = hms
                            u = ums
                            hu = h*u
                    else:
                        head = ul -sqrt(g*hl)
                        tail = umr - sqrt(g*hmr)
                        if x[i]<=head*t:
                            h = hl
                            u = ul
                            hu = h*u
                        elif x[i]>=tail*t:
                            h = hmr
                            u = umr
                            hu = h*u
                        else:
                            #inside the rarefaction
                            A = ul + 2*sqrt(g*hl)
                            h = (1/(9*g))*(A-(x[i]/t))**2
                            u = (x[i]/t) + sqrt(g*h)
                            hu = h*u
                   
                else:    
                    if hms>hr: #check if states are connected by shock wave
                        if x[i]<=t*sr(hms,qr,g):
                            h = hms
                            u = ums
                            hu = h*u
                        else:
                            h = hr
                            u =ur
                            hu = h*u
                    else: #rarefaction wave connection
                        head = ur + sqrt(g*hr)   #right eigen value
                        tail = umr + sqrt(g*hmr) #middle state eigen value
                        #check the region
                        if x[i] >= head*t: 
                            h = hr
                            u = ur
                            hu = h*u
                        elif x[i] <= tail*t: 
                            h = hmr
                            u = umr
                            hu = h*u
                        else:
                            #inside the rarefaction
                            A = ur - 2*sqrt(g*hr)
                            h = (1/(9*g))*(A-(x[i]/t))**2 
                            u = (x[i]/t) - sqrt(g*h)
                            hu = h*u
                q1[i] = h
                q2[i] = hu
                q3[i] = u
        
    if mq==0:
        return q1 #hieght field
    elif mq==1:
        return q2 #momentum field
    elif mq==3:
        return q3 #velocity field
#end qexact solver

#-------------------------------------------------------------------
# Exact Riemann solver that returns a single value at each interface 
# as it loops through all regions. 
# It is used by the approximate solvers to determine the intermediate
# state at each interface as the solution evolves.
# see brian_kyanjo_synthesis_duplicate.pdf for the mathematics
# behind the method. 
#-------------------------------------------------------------------
def exact(ql,qr,xi,mq,g):
    '''
    Description:
    -----------
    Uses the Initial Riemann problem to find an intemediate state (qm) which either 
    the intial left or right state connects to it via any combination of shocks and
    rarefactions in the two families.
    
    For Riemann problems with an initial dry state on one side, the exact Riemann 
    solution contains only a single rarefaction connecting the wet to dry state.

    The evolving wet dry interface is therefore simply one edge of the rarefaction. 
    The propagation speed of this interface can be exactly determined using the 
    Riemann invariants of the corresponding characteristics field.
    
    Input:
    -----
    x  - array of spacial points 
    t  - array of temporal points
    mq - specifies the output (0 and 1 corresponds to h and hu respectively) 
    ql - left initial state
    qr - right initial state
    Returns:
    h  - array of hieght field values
    hu - array of momentum field values

    '''
    
    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
    #dry velocity
    d_vl,d_vr = dry_velocity(ql, qr,g)
    
    #dry middle state
    if d_vl < d_vr:
        hmr = 0
        umr = 0.5*(d_vl + d_vr) #wet-dry interface speed (arbitrary)
        humr = hmr*umr
        
        #for i in range(len(x)):
        if xi<=lam1(hl,ul,g):
            h = hl
            u = ul
            hu = h*u
        elif lam1(hl,ul,g) <= xi and xi<=lam1(hmr,umr,g): 
            #inside the rarefaction
            A = ul + 2*sqrt(g*hl)
            h = (1/(9*g))*(A-(xi))**2
            u = (xi) + sqrt(g*h)
            hu = h*u
               
        elif lam1(hmr,umr,g)<=xi and xi<=lam2(hmr, umr,g):
            h = hmr
            hu = hmr*umr
            
        elif xi>=lam2(hmr, umr,g) and xi<=lam2(hr, ur,g):
            #inside the rarefaction
            A = ur - 2*sqrt(g*hr)
            h = (1/(9*g))*(A-(xi))**2                    
            u = (xi) - sqrt(g*h)
            hu = h*u
            
        else:
            h = hr
            hu = hr*ur
            
        #q1[i] = h
        #q2[i] = hu
            
    #dry left state (2-rarefaction only)
    elif hl == 0:
        hmr = 0
        umr = d_vr              #wet-dry interface speed
        humr = hmr*umr
    
        #for i in range(len(x)):
        if xi <= lam2(hmr, umr,g) :
            h = hl
            u = ul
            hu = h*u   
            
        elif xi>=lam2(hmr, umr,g) and xi<=lam2(hr, ur,g):
            #inside rarefaction
            A = ur - 2*sqrt(g*hr)
            h = (1/(9*g))*(A-(xi))**2
            u = (xi) - sqrt(g*h)
            hu = h*u
            
        else:
            h = hr
            u = ur
            hu = h*u
            
    #dry right state (1-rarefaction only)
    elif hr == 0:
        hmr = 0
        umr = d_vl              #wet-dry interface speed
        humr = hmr*umr
       
        #for i in range(len(x)):
        if xi<=lam1(hl,ul,g):
            h = hl
            u = ul
            hu = h*u
            
        elif lam1(hl,ul,g) <= xi and xi<=lam1(hmr,umr,g):
            #inside the rarefaction
            A = ul + 2*sqrt(g*hl)
            h = (1/(9*g))*(A-(xi))**2                   
            u = (xi) + sqrt(g*h)
            hu = h*u
            
        else:
            h = hr
            u = ur
            hu = h*u

    else:

        hs,us = Newton(ql,qr,g)
        
        if xi<=us:
            
            if hs>hl:
                s = ul - sqrt(0.5*g*hs/hl*(hl+hs))
                if xi<=s:
                    h = hl
                    hu = hl*ul
                else:
                    h = hs
                    hu = hs*us
            else:
                head = ul - sqrt(g*hl)
                tail = us - sqrt(g*hs)
                if xi <= head:
                    h = hl
                    hu = hl*ul
                elif xi>=tail:
                    h = hs
                    hu = hs*us
                else:
                    h = (((ul + 2*sqrt(g*hl) - xi)/3)**2)/g
                    u = xi + sqrt(g*h)
                    hu = h*u
        else:
            
            if hs>hr:
                s = ur + sqrt(0.5*g*hs/hr*(hs+hr))
                if xi<=s:
                    h = hs
                    hu = hs*us
                else:
                    h = hr
                    hu = hr*ur
            else:
                head = ur + sqrt(g*hr)
                tail = us + sqrt(g*hs)
                if (xi>=head):
                    h = hr
                    hu = hr*ur
                elif xi<=tail:
                    h = hs
                    hu = hs*us
                else:
                    h = (((xi-ur+2*sqrt(g*hr))/3)**2)/g
                    u = xi - sqrt(g*h)
                    hu = h*u
    if mq == 0:
        return h
    else:
        return hu

#end exact solver