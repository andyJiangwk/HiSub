'''Generating the peak mass function(PMF) for subhalos of any level'''
import numpy as np
from scipy.fft import fft,ifft
from scipy.interpolate import interp1d

def db(x,a1,a2,al1,al2,c,d):
    '''double-Schechter function'''
    return (a1*x**al1+a2*x**al2)*np.exp(-c*x**d)

def get_correct_parameter(fitting_tt,factor):
    ''' After applying the \beta factor for correction, the newly defined PMF in Equation 4 of Jiang et al.(2025) 
        is exactly the double-Schechter function with another set of paramters. 

        Therefore, this function is used to generate the parameters of corrected PMF for propagation.
    '''
    a1,a2,al1,al2,c,d = fitting_tt
    fitting_t1 = [0,0,0,0,0,0]
    fitting_t1[0] = a1*factor**(al1)/factor
    fitting_t1[1] = a2*factor**(al2)/factor
    fitting_t1[2] = al1
    fitting_t1[3] = al2
    fitting_t1[4] = c*factor**(d)
    fitting_t1[5] = d
    return fitting_t1

def level_o(x,n,init,conv):
    '''
    Functions that 


    Input
    --------------
    x: float array 
    An array of the interested peak mass ratio.

    n: int
    The hierarchical level of subhalo population
    (0 for all-level subhalos, n>1 for subhalos of specific level)

    init: function
    A function used to describe the level-1 PMF. 

    conv: function
    The convolution kernel to propagate the PMF into higher levels.
    

    Output:
    --------------
    The PMF values for the peak mass ratios input "x" of a given level n.
    '''
    if n==0:
        qspace = np.arange(-20,0,0.001)
        muspace = np.exp(qspace)
        yspace = np.array([init(np.exp(q1))*np.exp(q1) for q1 in qspace])
        yspace_t = np.array([conv(np.exp(q1))*np.exp(q1) for q1 in qspace])
        
        tk_t = fft(yspace)
        tk_tt = fft(yspace_t)
        tk_0 = tk_t/(1-tk_tt*abs(qspace[1]-qspace[0]))
        #tk_0 = tk_t/(1-tk_tt)
        y_inv0 = ifft(tk_0).real
        return intep1(x,muspace,y_inv0/muspace)
        
    if n>=1:
        qspace = np.arange(-50,1,0.001)
        muspace = np.exp(qspace)
        yspace = np.array([init(np.exp(q1))*np.exp(q1) for q1 in qspace])
        yspace_t = np.array([conv(np.exp(q1))*np.exp(q1) for q1 in qspace])
        N1 = len(qspace)
        ytest = []
        for i in range(n*N1-(n-1)):
            if i<=N1-1:
                ytest.append(yspace[i])
            else:
                ytest.append(0)

        ytest_t = []
        for i in range(n*N1-(n-1)):
            if i<=N1-1:
                ytest_t.append(yspace_t[i])
            else:
                ytest_t.append(0)
        
        ytest = np.array(ytest)
        ytest_t = np.array(ytest_t)
        
        tk_y = fft(ytest)
        tk_y_t = fft(ytest_t)
        y_inv2t = ifft((tk_y**1*tk_y_t**(n-1))).real*abs(qspace[1]-qspace[0])**(n-1)
        qs = np.arange(n*min(qspace),n*max(qspace)+0.0001,0.001)
        mus = np.exp(qs)
        #print(mus)
        return intep1(x,mus,y_inv2t/mus)


def intep1(x0,xlist,ylist):
    "Interpolation in log space."
    interp = interp1d(np.log10(xlist),np.log10(ylist))
    xres = []
    for x in x0:
        try:
            fac1 = np.log10(x)
            logx = interp(fac1)
            xres.append(10**(logx))
        except ValueError:
            xres.append(0)
    return np.array(xres)