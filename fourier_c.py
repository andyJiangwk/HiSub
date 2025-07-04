import numpy as np
from scipy.fft import fft,ifft
def db(x,a1,a2,al1,al2,c,d):
    return (a1*x**al1+a2*x**al2)*np.exp(-c*x**d)


def level_o(x,n,init,conv):
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

from scipy.interpolate import interp1d
def intep1(x0,xlist,ylist):
    from scipy.interpolate import interp1d
    interp = interp1d(np.log10(xlist),np.log10(ylist))
    xres = []
    for x in x0:
        try:
            fac1 = np.log10(x)
            logx = interp(fac1)
            xres.append(10**(logx))
        except ValueError:
            import traceback
            #traceback.print_exc()
            xres.append(0)
    return np.array(xres)