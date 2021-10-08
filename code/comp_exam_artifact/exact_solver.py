#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:08:44 2021

@author: Brian KYANJO
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:09:45 2021
@author: Brian KYANJO
"""

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
def newto(ql,qr,g):
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

#-------------------------------------------
#Another Newtow solver
def phi(hs,hlr,g):
    #lax-entropy condition
    if hs>hlr:
        #shock
        return (sqrt(0.5*g*(hs +hlr)/(hs*hlr))*(hs-hlr))
    else:
        #rarefaction
        return (2*sqrt(g)*(sqrt(hs) - sqrt(hlr)))

#find h*
def func(hs,hl,hr,ul,ur,g):
    return (phi(hs,hl,g) + phi(hs,hr,g) + ur - ul)

#derivative of func
def dfunc(hs,hl,hr,ul,ur,g):
    eps = 1e-7
    return (func(hs+eps,hl,hr,ul,ur,g) - func(hs-eps,hl,hr,ul,ur,g))/(2*eps)

#Newton solver
def newton(ql,qr,g):
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
    hs = ((sqrt(hl) + sqrt(hr) - (ur-ul)/2/sqrt(g))**2)/4
    
    tol = 1e-12
    max_iter = 100
    for i in range(max_iter):
        gk = func(hs,hl,hr,ul,ur,g) #fuction to be reset
        res = abs(gk)
        if res<tol:
            break
        else:
            continue
        
        dg = dfunc(hs,hl,hr,ul,ur,g)
        dh = -gk/dg  #Newton's step
        delta = 1 #scale factor 0<delta<1
        
        for j in range(1,20):
            if abs(func(hs+dh*delta,hl,hr,ul,ur,g)) >= res:
                delta = 0.5*delta #if the residue increases, reduce the Newton's step by one
            else:
                break #exit if the residue doesnt increase
        hs = hs + delta*dh
    
    us = ul - phi(hs,hl,g)
     
    return hs,us
#---------------------------------------------
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

#shock connection
def con_shock(ql,qr,g):
    '''
    Description: determines whether these two states can be connected by either a 1-shock or a 2-shock.
    Input: states ql and qr
    Output: determined shock and its corresponding speed
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
    
    #Intermediate state
    hm,um = newton(ql,qr,g)
    hum = hm*um
    
    if hm>hl and hm>hr:
        return 'all-shock'
    
    elif hm>hl:
        sl = (hul - hum)/(hl-hm)
        return '1-shock'
    
    elif hm>hr:
        sr = (hur - hum)/(hr-hm)
        return '2-shock'

#Rarefaction connection
def con_rare(ql,qr,g):
    '''
    Description: determines whether these two states can be connected by either a 1-shock or a 2-shock.
    -----------
    Input: states ql and qr
    -----
    Output: determined rarefaction and its corresponding speed
    ------
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
    
    #intermediate state
    hm,um = newton(ql,qr,g)
    hum = hm*um
    
    if hm<hl and hm<hr:
        return 'all-rarefaction'
    
    elif hm<hl:
        sl = (hul - hum)/(hl-hm) #shock speed
        return '1-rare'
    
    elif hm<hr:
        sr = (hur - hum)/(hr-hm) #shock speed
        return '2-rare'

#Dry velocity
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

#exact solution
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
    else:
        
        q1 = zeros(x.shape)
        q2 = zeros(x.shape)
                
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
        
        
        else:
        
            #for shock soln
            hms,ums = newto(ql,qr,g)
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
                            A = ul + 2*sqrt(g*hl)
                            h = (1/(9*g))*(A-(x[i]/t))**2
                            u = (x[i]/t) + sqrt(g*h)
                            hu = h*u
    
                    #q1[i] = h
                    #q2[i] = hu
                    
                else:
                    if hms>hr:
                        if x[i]<=t*sr(hms,qr,g):
                            h = hms
                            u = ums
                            hu = h*u
                        else:
                            h = hr
                            u =ur
                            hu = h*u
                    else:
                        head = ur + sqrt(g*hr)
                        
                        tail = umr + sqrt(g*hmr)
                        if x[i] >= head*t:
                            h = hr
                            u = ur
                            hu = h*u
                        elif x[i] <= tail*t:
                            h = hmr
                            u = umr
                            hu = h*u
                        else:
                            A = ur - 2*sqrt(g*hr)
                            h = (1/(9*g))*(A-(x[i]/t))**2 
                            u = (x[i]/t) - sqrt(g*h)
                            hu = h*u
                q1[i] = h
                q2[i] = hu
        
    if mq==0:
        return q1 #hieght field
    elif mq==1:
        return q2 #momentum field