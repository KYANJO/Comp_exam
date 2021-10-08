
from numpy import *

def pospart(x):
    '''
    Returns a value greater than zero
    '''
    return max(1e-15,x)


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
    #assert hl>=0, 'hl<0 {:.4e}'.format(hl)
    #assert hr>=0, 'hr<0 {:.4e}'.format(hr)
    d_vl = ul + 2*sqrt(g*hl)
    d_vr = ur - 2*sqrt(g*hr)
    
    return d_vl,d_vr

def exact(ql,qr,xi,mq,g):
    
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
        
        #q1[i] = h
        #q2[i] = hu   
    
            
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
            
        #q1[i] = h
        #q2[i] = hu

    else:

        hs,us = newton(ql,qr,g)
        #us = ul - phi(hs,hl,g)
        
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