from numpy import *

#shock wave solution
def newton(qo,ql,qr,g,max_iter,epislon):
    
    ho = qo[0]
    uo = qo[1]
    
    hl = ql[0]
    hul = ql[1]
    
    hr = qr[0]
    hur = qr[1]
    
    ul = hul/hl
    ur = hur/hr
    
    def f1(hm,um):
        return (um - (ur + (hm-hr)*sqrt((g/2)*(1/hm + 1/hr))))
    
    def f2(hm,um):
        return (um - (ul - (hm-hl)*sqrt((g/2)*(1/hm + 1/hl))))

    #def f2(hm,um):
     #   return (um - (ul + 2*(sqrt(g*hl) - sqrt(g*hm))))

    #Derivatives
    def f1h(hm,um):
        return (sqrt(2*g*(hm + hr)/(hm*hr)))*(-2*hm*(hm + hr) + hr*(hm - hr)) \
              / (4*hm*(hm + hr))

    def f1u(hm,um):
        return 1

    def f2h(hm,um):
        return (sqrt(2*g*(hm + hl)/(hm*hl)))*(2*hm*(hm + hl) + hl*(hl - hm)) \
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

        if linalg.norm(v1-vo) < epislon:
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

    ul = hul/hl
    ur = hur/hr

    hm = (1/16*g)*(ul - ur + 2*(sqrt(g*hl) + sqrt(g*hr)))**2
    um = ul + 2*(sqrt(g*hl) - sqrt(g*hm))
    return hm,um

#shock speed
#location of the shock
def sl(h,ql,g):
    
    hl = ql[0]
    hul = ql[1]
    
    ul = hul/hl
    
    return (ul - (1/hl)*sqrt((g/2)*(hl*h*(hl+h))))

def sr(h,qr,g):
    
    hr = qr[0]
    hur = qr[1]
    
    ur = hur/hr
    
    return (ur + (1/hr)*sqrt((g/2)*(hr*h*(hr+h))))

#eigen values
def lam1(h,u,g):
    return (u - sqrt(g*h))

def lam2(h,u,g):
    return (u + sqrt(g*h))

#exact solution
def qexact(x,t,mq,ql,qr,qmr,qms,lam1,g):
    
    hl = ql[0]
    hul = ql[1]
    
    hr = qr[0]
    hur = qr[1]
    
    ul = hul/hl
    ur = hur/hr
    
    hmr = qmr[0]
    humr = qmr[1]
    
    hms= qms[0]
    hums = qms[1]
    
    umr = humr/hmr
    ums = hums/hms
    
    if t==0:
        q1 = where(x<0,hl,hr)
        q2 = where(x<0,hl*ul,hr*ur)
    else:
        q1 = zeros(x.shape)
        q2 = zeros(x.shape)
        
        L = [lam1,lam2]
        for i in range(len(x)):
            for k in range(len(L)):
                if L[k](hl,ul,g)<L[k](hr,ur,g):
                    if x[i]<L[k](hl,ul,g)*t:
                        h = hl
                        hu = hl*ul
                    elif L[k](hl,ul,g)*t <= x[i] and x[i]<L[k](hmr,umr,g)*t:
                        #inside the rarefaction
                        if k==0:
                            A = ul + 2*sqrt(g*hl)
                            h = (1/(9*g))*(A-(x[i]/t))**2
                            u = ul + 2*(sqrt(g*hl) - sqrt(g*h)) 
                            #u = (x[i]/t) + sqrt(g*h)
                            hu = h*u
                        else:
                            A = ur - 2*sqrt(g*hr)
                            h = (1/(9*g))*(A-(x[i]/t))**2
                            u = ur - 2*(sqrt(g*hr) - sqrt(g*h)) 
                            #u = (x[i]/t) - sqrt(g*h)
                            hu = h*u
                            
                else:
                    if L[k](hmr,umr,g)*t< x[i] and x[i]<t*sr(hms,qr,g):
                        h = hms
                        hu = hms*ums
                    else:
                        h = hr
                        hu = hr*ur
                q1[i] = h
                q2[i] = hu
            
    if mq==0:
        return q1
    elif mq==1:
        return q2

    
