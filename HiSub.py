import numpy as np
import math
import matplotlib.pyplot as plt
import math
from fourier_c import db,level_o,get_correct_parameter

class HiSub:
    '''
        The model used to analytically desribe the hierarchical origin of subhalos at different levels.

        Input:
        -------------
            fitting_param: float array
                The fitting param for the level 1 PMF in the form of (μdN/dlnμ)
            beta: float 
                The correction factor
    '''
    def __init__(self,fitting_param,beta):
        self.fitting_param_init = fitting_param
        self.beta          = beta
        self._trans_param()
        
    def initial_merger_ratio(self,xi_array,*,mu_0,depth):
        '''
            Calculating the initial merger ratio distribution for subhalos with given final peak mass ratio 
            and levels. Details can be found in Equation 14 in Jiang et al.(2025)

            Input:
            -------------
                xi_array: float array
                    An array of specified iniital merger ratio

                mu_0:     float
                    The final peak mass ratio for a subhalo

                depth:    float 
                    The hierarchical level of a subhalo

            Output:
            -------------
                z_distribution: float array
                    An array describing the dp/dxi distribution with the initial merger ratio specificed by xi_array
        '''
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
        distribution = reslist/norm
        return distribution
    
    def accretion_rate(self,mu_array,depth):
        '''
            Calculating the specific accretion rate for subhalos at a level with given depth. 
            (Equation 12 in Jiang et al.(2025))

            Input:
            -------------
                mu_array: float array 
                    An array specified the mass ratio between subhalos and their host halos at the accretion time.

                depth: int 
                    The hierarchical level of the subhalo population

            Output:
            -------------
                derivative: float array
                    An array that showing the specific accretion rate of subhalos of a given level with given accretion mass ratios.
        '''
        if isinstance(mu_array,np.ndarray):
            mu_array1 = mu_array*(1+0.001)
            st1 = self._PMF1(mu_array,depth)*mu_array
            st2 = self._PMF1(mu_array1,depth)*mu_array1
            direvative = -(st2-st1)/(mu_array1-mu_array)
            return direvative
    
    def accretion_redshift(self,mu_0,z_array,depth,*,MAH,M_halo):
        '''
            Calculating the accretion redshifts distribution for subhalo population of given level with given final peak mass ratio.
            (Equation 16 in Jiang et al.(2025))

            Input:
            -------------
                mu_0: float
                    The final peak mass ratio of the subhalo population.

                z_array: float list
                    An array of redshifts in ascending order representing the interested redshift ranges.
                
                depth: int
                    The hierarchical level of given subhalo population.
                
                MAH: class
                    The mass assembly history model. It should be a class at least containing the member method MAH.get_growth,MAH.get_growth_rate.
                    The former calculates the mass at different redshifts for halos with given mass. The latter calculates the derivatives of the growth mass with respect
                    to the redshifts. 
                    
                    One example has been shown in the MAH.py which is developed from Zhao et al.(2008)
                
                M_halo: int
                    The mass of the halo unit: 10^10 Msun

            Output:
            -------------
                z_distribution: float array
        '''
        Mhistory = MAH.get_growth(M_halo,z_array)
        Mhat = Mhistory/M_halo
        Macc_rate = MAH.get_growth_rate(Mhistory,z_array,z_array[0],Mhistory[0],div_fac)/Mhistory[0]

        f_2 = self.accretion_rate(mu_0/Mhat,depth)
        PMF_norm = self._PMF1(np.array([mu0,]),depth) 

        fac1 = -(f_2/Mhat**2)
        fac2 = Macc_rate/PMF_norm
        z_distribution = fac1*fac2
        return z_distribution
    
    def _trans_param(self):
        self.fitting_param_conv = get_correct_parameter(self.fitting_param_init,self.beta)
    
    def _PMF1(self,x,depth):
        ''' PMF in the form of dN/dμ'''
        return level_o(x,depth,lambda x: db(x,*self.fitting_param_init),lambda x: db(x,*self.fitting_param_conv))/x**2
    
    def _PMF2(self,x):
        ''' Level 1 convolved PMF in the form of dN/dμ'''
        return db(x,*self.fitting_param_conv)