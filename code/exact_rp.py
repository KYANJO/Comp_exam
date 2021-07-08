#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:09:45 2021
@author: Brian KYANJO
"""


from numpy import *


def pospart(x):
    return max(1e-15,x)

#shock wave solution
def newton(ql,qr,g):
    
    #global g
    
    max_iter = 100
    epsilon  = 1e-16
    
    ho = 0.1
    uo = 0.01
    
    hl = ql[0]
    hul = ql[1]
    
    hr = qr[0]
    hur = qr[1]
    
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
    def f1(hm,um):
        return (um - (ur + (hm-hr)*sqrt((g/2)*(1/hm + 1/pospart(hr)))))
    
    def f2(hm,um):
        return (um - (ul - (hm-hl)*sqrt((g/2)*(1/hm + 1/pospart(hl)))))

    #def f2(hm,um):
     #   return (um - (ul + 2*(sqrt(g*hl) - sqrt(g*hm))))

    #Derivatives
    def f1h(hm,um):
        return (sqrt(2*g*(hm + hr)/(hm*pospart(hr))))*(-2*hm*(hm + hr) + hr*(hm - hr)) \
              / (4*hm*(hm + hr))

    def f1u(hm,um):
        return 1

    def f2h(hm,um):
        return (sqrt(2*g*(hm + hl)/(hm*pospart(hl))))*(2*hm*(hm + hl) + hl*(hl - hm)) \
              / (4*hm*(hm + hl))
    #def f2h(hm,um):
     #   return ((sqrt(g*hm))/hm)

    def f2u(hm,um):
        return 1

    #Jacobian
    def J(f1h,f1u,f2h,f2u,hm,um):
        return array([[f1h(hm,um),f1u(hm,um)],[f2h(hm,um),f2u(hm,um)]])

    #inverse of J
    def Jinv(hm,um):
        return linalg.inv(J(f1h,f1u,f2h,f2u,hm,um))

    def f(hm,um):
        return array([f1(hm,um),f2(hm,um)])

    #intial value

    vo = array([ho,uo])

    #method
    for i in range(max_iter):

        v1 = vo - Jinv(ho,uo)@f(ho,uo)

        if linalg.norm(v1-vo) < epsilon:
            break
        else:
            vo = v1
            ho = v1[0]
            uo = v1[1]
            
    return v1[0],v1[1]*v1[0]

#Rarefaction wave solution
def rare(ql,qr,g):

    hl = ql[0]
    hul = ql[1]

    hr = qr[0]
    hur = qr[1]

    ul = hul/pospart(hl)
    ur = hur/pospart(hr)

    hm = (1/16*g)*(ul - ur + 2*(sqrt(g*hl) + sqrt(g*hr)))**2
    um = ul + 2*(sqrt(g*hl) - sqrt(g*hm))
    return hm,um

#shock speed
#location of the shock
def sl(h,ql,g):
    
    hl = ql[0]
    hul = ql[1]
    
    ul = hul/pospart(hl)
    
    return (ul - (1/pospart(hl))*sqrt((g/2)*(hl*h*(hl+h))))

def sr(h,qr,g):
    
    hr = qr[0]
    hur = qr[1]
    
    ur = hur/pospart(hr)
    
    return (ur + (1/pospart(hr))*sqrt((g/2)*(hr*h*(hr+h))))

#eigen values
def lam1(h,u,g):
    return (u - sqrt(g*h))

def lam2(h,u,g):
    return (u + sqrt(g*h))

#shock connection
def con_shock(ql,qr,g):
    '''
    Description: determines whether these two states can be connected by either a 1-shock or a 2-shock.
    Input: states ql and qr
    Output: determined shock and its corresponding speed
    '''
    hl = ql[0]
    hr = qr[0]
    
    hul = ql[1]
    hur = qr[1]
    
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
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
    Input: states ql and qr
    Output: determined rarefaction and its corresponding speed
    '''
    hl = ql[0]
    hr = qr[0]
    
    hul = ql[1]
    hur = qr[1]
    
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
    hm,um = newton(ql,qr,g)
    hum = hm*um
    
    if hm<hl and hm<hr:
        return 'all-rarefaction'
    
    elif hm<hl:
        sl = (hul - hum)/(hl-hm)
        return '1-rare'
    
    elif hm<hr:
        sr = (hur - hum)/(hr-hm)
        return '2-rare'

#Dry velocity
def dry_velocity(ql,qr,g):
    hl = ql[0]
    hr = qr[0]
    
    hul = ql[1]
    hur = qr[1]
    
    ul = hul/pospart(hl)
    ur = hur/pospart(hr)
    
    d_vl = ul + 2*sqrt(g*hl)
    d_vr = ur - 2*sqrt(g*hr)
    
    return d_vl,d_vr

#exact solution
def qexact(x,t,mq,ql,qr,g):
    
    hl = ql[0]
    hr = qr[0]
    
    hul = ql[1]
    hur = qr[1]
    
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
            umr = 0.5*(d_vl + d_vr) #wet-dry interface speed
            #umr = 0
            humr = hmr*umr
            
            for i in range(len(x)):
                if x[i]<lam1(hl,ul,g)*t:
                    h = hl
                    u = ul
                    hu = h*u
                elif lam1(hl,ul,g)*t <= x[i] and x[i]<lam1(hmr,umr,g)*t: 
                    #inside the rarefaction
                    A = ul + 2*sqrt(g*hl)
                    h = (1/(9*g))*(A-(x[i]/t))**2
                    #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                    u = (x[i]/t) + sqrt(g*h)
                    hu = h*u
                       
                elif lam1(hmr,umr,g)*t<=x[i] and x[i]<=t*lam2(hmr, umr,g):
                    h = hmr
                    hu = hmr*umr
                    
                elif x[i]>t*lam2(hmr, umr,g) and x[i]<t*lam2(hr, ur,g):
                    #inside the rarefaction
                    A = ur - 2*sqrt(g*hr)
                    h = (1/(9*g))*(A-(x[i]/t))**2
                    #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
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
                if x[i] < t*lam2(hmr, umr,g) :
                    h = hl
                    u = ul
                    hu = h*u   
                    
                elif x[i]>t*lam2(hmr, umr,g) and x[i]<t*lam2(hr, ur,g):
                    #inside rarefaction
                    A = ur - 2*sqrt(g*hr)
                    h = (1/(9*g))*(A-(x[i]/t))**2
                    #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
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
            humr =hmr*umr
           
            for i in range(len(x)):
                if x[i]<lam1(hl,ul,g)*t:
                    h = hl
                    u = ul
                    hu = h*u
                    
                elif lam1(hl,ul,g)*t <= x[i] and x[i]<lam1(hmr,umr,g)*t:
                    #inside the rarefaction
                    A = ul + 2*sqrt(g*hl)
                    h = (1/(9*g))*(A-(x[i]/t))**2
                    #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
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
            hms,ums = newton(ql,qr,g)
            hums = hms*ums
            qms = array([hms,hums])
            
            #for rarefaction soln
            hmr,umr = rare(ql,qr,g)
            humr = hmr*umr
            qmr = array([hmr,humr])  
            
            #All rarefaction
            if con_rare(ql,qr,g) == 'all-rarefaction':
                
                for i in range(len(x)):
                    if x[i]<lam1(hl,ul,g)*t:
                        h = hl
                        u = ul
                        hu = h*u
                    elif lam1(hl,ul,g)*t <= x[i] and x[i]<lam1(hmr,umr,g)*t: 
                        #inside the rarefaction
                        A = ul + 2*sqrt(g*hl)
                        h = (1/(9*g))*(A-(x[i]/t))**2
                        #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                        u = (x[i]/t) + sqrt(g*h)
                        hu = h*u
                           
                    elif lam1(hmr,umr,g)*t<=x[i] and x[i]<=t*lam2(hmr, umr,g):
                        
                        h = hmr
                        hu = hmr*umr
                    elif x[i]>t*lam2(hmr, umr,g) and x[i]<t*lam2(hr, ur,g):
                        #inside the rarefaction
                        A = ur - 2*sqrt(g*hr)
                        h = (1/(9*g))*(A-(x[i]/t))**2
                        #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                        u = (x[i]/t) - sqrt(g*h)
                        hu = h*u
                        
                    else:
                        h = hr
                        hu = hr*ur
                        
                    q1[i] = h
                    q2[i] = hu
                    
            #Dam break (1-rarefaction and 2-shock)
            elif con_rare(ql,qr,g) == '1-rare' and \
                 con_shock(ql,qr,g) == '2-shock':    
    
                for i in range(len(x)):
                    if x[i]<lam1(hl,ul,g)*t:
                        h = hl
                        u = ul
                        hu = h*u
                        
                    elif lam1(hl,ul,g)*t <= x[i] and x[i]<lam1(hmr,umr,g)*t:
                        #inside the rarefaction
                        A = ul + 2*sqrt(g*hl)
                        h = (1/(9*g))*(A-(x[i]/t))**2
                        #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                        u = (x[i]/t) + sqrt(g*h)
                        hu = h*u
                        
                    elif lam1(hmr,umr,g)*t< x[i] and x[i]<t*sr(hms,qr,g):
                        h = hmr
                        u = umr
                        hu = h*u
                        
                    else:
                        h = hr
                        u = ur
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
                    
             #All shock        
            elif con_shock(ql,qr,g) == 'all-shock':
                
                for i in range(len(x)):
                    if 0.5*(sl(hl,ql,g) + sl(hms,qms,g))*t > x[i] :
                        h = hl
                        u = ul
                        hu = h*u   
                    
                    elif sl(hms,qms,g)*t < x[i] and x[i]<t*sr(hms,qr,g):
                        h = hms
                        u = ums
                        hu = h*u
                        
                    else:
                        h = hr
                        u = ur
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
                    
            #1-shock and 2-rarefaction    
            elif con_shock(ql,qr,g) == '1-shock' and \
                 con_rare(ql,qr,g) == '2-rare':
                
                for i in range(len(x)):
                    if 0.5*(sl(hl,ql,g) + sl(hms,qms,g))*t > x[i] :
                        h = hl
                        u = ul
                        hu = h*u   
                    
                    elif sl(hms,qms,g)*t < x[i] and x[i]<=t*lam2(hmr, umr,g):
                        h = hmr
                        u = umr
                        hu = h*u
                        
                    elif x[i]>t*lam2(hmr, umr,g) and x[i]<t*lam2(hr, ur,g):
                        #inside rarefaction
                        A = ur - 2*sqrt(g*hr)
                        h = (1/(9*g))*(A-(x[i]/t))**2
                        #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                        u = (x[i]/t) - sqrt(g*h)
                        hu = h*u
                        
                    else:
                        h = hr
                        u = ur
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
            
            
            #1-shock only (left going shock)
            elif con_shock(ql,qr,g) == '1-shock':
                for i in range(len(x)):
                    if 0.5*(sl(hl,ql,g) + sl(hms,qms,g))*t > x[i] :
                        h = hl
                        u = ul
                        hu = h*u   
                    
                    else:
                        h = hms
                        u = ums
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
                    
            #2-shock only (right going shock)    
            elif con_shock(ql,qr,g) == '2-shock':   
                for i in range(len(x)):
                    if x[i]<=t*sr(hms,qr,g):
                        h = hms
                        u = ums
                        hu = h*u
                        
                    else:
                        h = hr
                        u = ur
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
                
            #1-rarefaction only (left going rarefaction)    
            elif con_rare(ql,qr,g) == '1-rare':
                for i in range(len(x)):
                    if x[i]<lam1(hl,ul,g)*t:
                        h = hl
                        u = ul
                        hu = h*u
                        
                    elif lam1(hl,ul,g)*t <= x[i] and x[i]<lam1(hmr,umr,g)*t:
                        #inside the rarefaction
                        A = ul + 2*sqrt(g*hl)
                        h = (1/(9*g))*(A-(x[i]/t))**2
                        #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                        u = (x[i]/t) + sqrt(g*h)
                        hu = h*u
                        
                    else:
                        h = hr
                        u = ur
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
                
            #2-rarefaction only (right going rarefaction)
            elif con_rare(ql,qr,g) == '2-rare':
                for i in range(len(x)):
                    if x[i] < t*lam2(hmr, umr,g) :
                        h = hl
                        u = ul
                        hu = h*u   
                        
                    elif x[i]>t*lam2(hmr, umr,g) and x[i]<t*lam2(hr, ur,g):
                        #inside rarefaction
                        A = ur - 2*sqrt(g*hr)
                        h = (1/(9*g))*(A-(x[i]/t))**2
                        #u = umr + 2*(sqrt(g*hmr) - sqrt(g*h)) 
                        u = (x[i]/t) - sqrt(g*h)
                        hu = h*u
                        
                    else:
                        h = hr
                        u = ur
                        hu = h*u
                        
                    q1[i] = h
                    q2[i] = hu
        
    if mq==0:
        return q1
    elif mq==1:
        return q2