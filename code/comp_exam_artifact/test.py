#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 8 03:35:43 2021

@author: Brian Kyanjo
"""
#-------------------------------------------------------------
#Test script 
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

from numpy import *
import exact_solver  #Contains the exact solvers
import approximate_solver #Contains approximate solvers
from matplotlib.pylab import *
from sys import exit
import warnings
warnings.filterwarnings('ignore')

# Initial conditions
#The solver should supply an initialization routine to initialize  q(x,t)  at time  t=0 .
def h_init(x,hl,hr):    
    q0 = where(x < 0,hl,hr)
    return q0

def hu_init(x,hl,ul,hr,ur):    
    #q0 = zeros(x.shape)  
    q0 = where(x<0,hl*ul,hr*ur)
    return q0

def qinit(x,meqn,ql,qr):
    #initial height fields(left and right)
    hl = ql[0]
    hr = qr[0]
    
    #initial momentum fields(left and right)
    hul = ql[1]
    hur = qr[1]
    
    #initial momentum fields(left and right)
    ul = hul/exact_solver.pospart(hl)
    ur = hur/exact_solver.pospart(hr)

    q = zeros((x.shape[0],meqn))
    q[:,0] = h_init(x,hl,hr)
    q[:,1] = hu_init(x,hl,ul,hr,ur)
    
    return q

# Boundary conditions
def bc_extrap(Q):
    """ Extend Q with extrapolation boundary conditions """
        
    Q_ext = concatenate((Q[[1,0],:], Q, Q[[-1,-2],:]))
    return Q_ext

# Problem test
def problem_test(case,itype):
    '''
    Description: Contains all test cases used in this simulation including wet dry state cases.
    input: case, itype
    output: outputs left (ql) and right states (qr)
    '''
    if itype == 1 and case > 7:
        print('The approximate solves can\'t handle dry states yet, please choose itype = 0 for presence of dry states.')
        exit()
    elif itype == 2 and case > 7:
        print('The approximate solves can\'t handle dry states yet, please choose itype = 0 for presence of dry states.')
        exit()

    elif case == 0:     #left going shock
        hl = 1
        hr = 1.5513875245483204
        ul = 0.5
        ur = 0
        print('Problem: left going shock\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    elif(case == 1):  #right going shock
        hl = 1.5513875245483204
        hr = 1
        ul = 0.0
        ur = -0.5
        print('Problem: right going shock\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    elif(case == 2):  #right going rarefaction
        hl = 0.5625
        hr = 1
        ul = 0
        ur = 0.5
        print('Problem: right going rarefaction\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    elif(case == 3):  #left going rarefaction
        hl = 2
        hr = 1.4571067811865475
        ul = 0
        ur = 0.41421356237309537
        print('Problem: left going rarefaction')

    elif(case == 4): #dam break
        hl = 2
        hr = 1
        ul = 0
        ur = 0
        print('Problem: dam break problem \n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    elif (case == 5): #All rarefaction 
        hl = 1
        hr = 1
        ul = -0.5
        ur = 0.5
        print('Problem test: All rarefaction\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)
    elif (case == 6): #All shock 
        hl = 1
        hr = 1
        ul = 0.5
        ur = -0.5
        print('Problem: All shock\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)
    #elif itype == 1: #presence of dry states 
    elif case == 7: #left dry state
        hl = 0
        ul = 0
        hr = 1
        ur = 0
        print('Problem: Left dry state\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    elif case == 8: #middle dry state
        hl = .1
        ul = -.7
        hr = .1
        ur = 0.7
        print('Problem: middle dry state\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    elif case == 9: #right dry state
        hl = 1
        ul = 0
        hr = 0
        ur = 0
        print('Problem: right dry state\n','\t hl = ', hl,'\n',\
            '\t hr = ', hr,'\n','\t ul = ', ul,'\n','\t ur = ', ur)

    ql = array([hl,hl*ul])
    qr = array([hr,hr*ur])

    return ql,qr

#plot function
def Riemansoln(umax,to,mq,case,itype,g,ax,bx, ay,by,mx, Tfinal, \
    limiter_choice,second_order,meqn,solver,cfl):
    '''
    Description: calls the exact and approximate solvers and then returns 
                 plots of Riemann solutions for both standard cases and 
                 presence of dry states cases.
    '''
    #spatial mesh
    dx = (bx-ax)/mx
    xe = linspace(ax,bx,mx+1)  # Edge locations
    xc = xe[:-1] + dx/2       # Cell-center locations

    # Estimate time step and number of time steps to take
    dt_est = cfl*dx/umax
    nout = int(floor(Tfinal/dt_est) + 1) #number of time steps
    dt = Tfinal/nout
    tvec = linspace(to,Tfinal,nout+1)

    #sample problem test
    ql,qr = problem_test(case,itype)

    #sample graphics
    #only exact solver ploted due to presence of dry states
    if itype == 0:
        fig = figure(1)
        clf()
        #initialise the exact soln
        qeo = exact_solver.qexact(xc,to,mq,ql,qr,g)
        hde, = plot(xc,qeo,'r-',markersize=5,label='dry_wet')

        if mq == 0:
            tstr = 'Height : t = {:.4f}'
        else:
            tstr = 'Momentum : t = {:.4f}'

        htitle = title(tstr.format(0),fontsize=18)
        #grid()

        for i,t in enumerate(tvec):

            #exact solution
            qe = exact_solver.qexact(xc,t,mq,ql,qr,g)
            hde.set_ydata(qe)

            xlabel('x',fontsize=16)
            ylabel('q(x,t)',fontsize=16)
            htitle.set_text(tstr.format(t))

            legend()

            ylim([ay,by])

            pause(0.1)
            #savefig('/Users/mathadmin/Documents/phd-research/comprehensive_exam/synthesis_paper/images/rr')

            fig.canvas.draw()        

    # Selected approximate solver is compared with the exact solver
    elif itype == 1:
        Q,xc,tvec = approximate_solver.claw(ax,bx, mx,  Tfinal, nout,ql,qr, \
                  meqn=meqn, \
                  exact=exact_solver.exact,\
                  solver=solver, \
                  qinit=qinit, \
                  bc=bc_extrap, \
                  limiter_choice=limiter_choice,
                  second_order=second_order)

        fig = figure(1)
        clf()

        #initialise the exact soln
        qeo = exact_solver.qexact(xc,to,mq,ql,qr,g)
        hde, = plot(xc,qeo,'r-',markersize=5,label='Exact')

        q0 = Q[:,mq,0]
        hdl, = plot(xc,q0,'b.',markersize=5,label='Approximated')

        if mq == 0:
            tstr = 'Height : t = {:.4f}'
        else:
            tstr = 'Momentum : t = {:.4f}'

        htitle = title(tstr.format(0),fontsize=18)

        for i,t in enumerate(tvec):

            #exact solution
            qe = exact_solver.qexact(xc,t,mq,ql,qr,g)
            hde.set_ydata(qe)

            #wpa
            q = Q[:,mq,i]
            hdl.set_ydata(q)

            xlabel('x',fontsize=16)
            ylabel('q(x,t)',fontsize=16)
            htitle.set_text(tstr.format(t))

            legend()

            ylim([ay,by])

            pause(0.1)
            #savefig('/Users/mathadmin/Documents/phd-research/comprehensive_exam/synthesis_paper/images/rr')

            fig.canvas.draw()  

    #All approximate solvers compared with the exact solver
    elif itype == 2:
        fig = figure(1)
        clf()
       
        #Roe-solver
        Qr,xc,tvec = approximate_solver.claw(ax,bx, mx,  Tfinal, nout,ql,qr, \
                meqn=meqn, \
                exact=exact_solver.exact,\
                solver=0, \
                qinit=qinit, \
                bc=bc_extrap, \
                limiter_choice=limiter_choice,
                second_order=second_order)

        #flux-wave decomposition solver
        Qfl,xc,tvec = approximate_solver.claw(ax,bx, mx,  Tfinal, nout,ql,qr, \
                meqn=meqn, \
                exact=exact_solver.exact,\
                solver=1, \
                qinit=qinit, \
                bc=bc_extrap, \
                limiter_choice=limiter_choice,
                second_order=second_order)

        #fwave based solver
        Qfw,xc,tvec = approximate_solver.claw(ax,bx, mx,  Tfinal, nout,ql,qr, \
                meqn=meqn, \
                exact=exact_solver.exact,\
                solver=2, \
                qinit=qinit, \
                bc=bc_extrap, \
                limiter_choice=limiter_choice,
                second_order=second_order)

        #initialise the approximate soln with Roe-solver
        qroe = Qr[:,mq,0]
        hdr, = plot(xc,qroe,'.',markersize=5,label='$Roe_{solver}$')

        #initialise the approximate soln with flux decomposition-solver
        qfl = Qfl[:,mq,0]
        hdfl, = plot(xc,qfl,'*-',markersize=5,label='$flux_{solver}$')

        #initialise the approximate soln with fwave-solver
        qfw = Qfw[:,mq,0]
        hdfw, = plot(xc,qfw,':',markersize=5,label='$fwave_{solver}$')

        #initialise the exact soln
        qeo = exact_solver.qexact(xc,to,mq,ql,qr,g)
        hde, = plot(xc,qeo,'--',markersize=5,label='$Exact_{solver}$')

        if mq == 0:
            tstr = 'Height : t = {:.4f}'
        else:
            tstr = 'Momentum : t = {:.4f}'

        htitle = title(tstr.format(0),fontsize=18)

        for i,t in enumerate(tvec):

            #exact solution
            qe = exact_solver.qexact(xc,t,mq,ql,qr,g)
            hde.set_ydata(qe)

            #Roe
            qRoe = Qr[:,mq,i]
            hdr.set_ydata(qRoe)

            #flux decomposition
            qfl = Qfl[:,mq,i]
            hdfl.set_ydata(qfl)

            #fwave
            qfw = Qfw[:,mq,i]
            hdfw.set_ydata(qfw)

            xlabel('x',fontsize=16)
            ylabel('q(x,t)',fontsize=16)
            htitle.set_text(tstr.format(t))

            legend()

            ylim([ay,by])

            pause(0.1)
            #savefig('/Users/mathadmin/Documents/phd-research/comprehensive_exam/synthesis_paper/images/rr')

            fig.canvas.draw()                  