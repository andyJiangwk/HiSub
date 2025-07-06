import numpy as np
import math
import matplotlib.pyplot as plt
import math
from fourier_c import db,level_o,get_correct_parameter

class HiSub:
    def __init__(self,fitting_param,beta):
        self.fitting_param_init = fitting_param
        self.beta          = beta
        self._trans_param()
        
    def initial_merger_ratio(self,xi_array,*,mu_0,depth):
        PMF_norm = self._PMF1(np.array([mu_0,]),depth)
        xi_t = mu_0/xi_array
        y_t1 = self._PMF1(xi_t,depth-1)
        y_t2 = self._PMF2(xi_array)
        reslist = y_t1*y_t2/xi/PMF_norm
        reslist=np.array(reslist)

        norm = 0
        for dx,dp in zip(np.diff(xi),(reslist[1:]+reslist[:-1])/2):
            if np.isnan(dp):
                continue
            else:
                norm += dp*dx
        return reslist/norm
    def accretion_rate(self,mu_array,depth):
        if isinstance(mu_array,np.ndarray):
            mu_array1 = mu_array*(1+0.001)
            st1 = self._PMF1(mu_array,depth)*mu_array
            st2 = self._PMF1(mu_array1,depth)*mu_array1
            direvative = -(st2-st1)/(mu_array1-mu_array)
            return direvative
        '''
        else:
            mu_array = np.array([mu_array,])
            mu_array1 = mu_array*(1+0.001)
            st1 = _PMF1(mu_array,depth)*mu_array
            st2 = _PMF1(mu_array1,depth)*mu_array1
            direvative = -(st2-st1)/(mu_array1-mu_array)
            return derivative[0]
        '''
    def accretion_redshift(self,mu_0,z_array,depth,*,MAH,M_halo):
        Mhistory = MAH.get_growth(M_halo,z_array)
        Mhat = Mhistory/M_halo
        Macc_rate = MAH.get_accf_nop(Mhistory,z_array,z_array[0],Mhistory[0],div_fac)/Mhistory[0]

        f_2 = self.accretion_rate(mu_0/Mhat,depth)
        PMF_norm = self._PMF1(np.array([mu0,]),depth) 

        fac1 = -(f_2/Mhat**2)
        fac2 = Macc_rate/PMF_norm
        return fac1*fac2
    
    def _trans_param(self):
        self.fitting_param_conv = get_correct_parameter(self.fitting_param_init,self.beta)
    
    def _PMF1(self,x,depth):
        return level_o(x,depth,lambda x: db(x,*self.fitting_param_init),lambda x: db(x,*self.fitting_param_conv))/x**2
    
    def _PMF2(self,x):
        return db(x,*self.fitting_param_conv)