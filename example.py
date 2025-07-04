'''an example code to calculate the peak mass function for subhalos of any level.'''
import numpy as np
import matplotlib.pyplot as plt
from fourier_c import level_o
from fourier_c import db

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


cols = ['g','r','b','orange','purple','brown','cyan','y','pink']
ft_tick=22
ft_text=33
xtest = np.logspace(-4,0,100)

fitting_t = (0.029, 0.273, 1-0.94, 1-0.54, 12.89, 2.26) # The level-1 PMF is described by the double-Schechter function, and in the form of  (\mu)*dN/dlog(\mu)
fitting_t1 = get_correct_parameter(fitting_t,0.726) # The convolutional kernel

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot()

for i in range(4):
    depth = i+1
    ytest = level_o(xtest,depth,lambda x:db(x,*fitting_t),lambda x:db(x,*fitting_t1)) #Calculate the PMFs at first four levels
    ax1.plot(xtest,ytest,c=cols[i],linestyle='--',lw=3,alpha=0.6,label='$level\;%d$'%(depth)) 

y_t_all = ytest = level_o(xtest,0,lambda x:db(x,*fitting_t),lambda x:db(x,*fitting_t1)) #Calculate the PMFs for all-level subhalos.
ax1.plot(xtest,y_t_all,c='k',linestyle='-',lw=3,alpha=0.4,label='all')

#Basic Elements in the plot
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(1e-3,7e-1)
ax1.set_xlim(1e-4,1)
ax1.set_xlabel(r'$\mu$',fontsize=44)
ax1.set_ylabel(r'$\mu {\rm d}N/{\rm dln}\mu$',fontsize=44)
ax1.legend(loc='upper left',frameon=0)