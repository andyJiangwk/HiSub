import numpy as np
import math
from colossus.cosmology import cosmology

class MAH_Zhao:
    '''
        Model for the mass assembly history(MAH) of given halos.
        It is developed based on the formulas in Zhao et al.(2008)
    '''
    def __init__(self, H=100, G=43.007):
        self.params = {
            'flat': True,
            'H0': 67.8,
            'Om0': 0.268,
            'Ob0': 0.049,
            'sigma8': 0.863,
            'ns': 0.963
        }
        self.cosmo = cosmology.setCosmology('myCosmo', **self.params)
        self.H = H
        self.G = G
        self.rho = 3 * H**2 / (8 * math.pi * G) * self.params['Om0']

    def _sigma(self, M):
        '''Calculate \sigma(M)'''
        r0 = self._R(M)
        return self.cosmo.sigma(r0, z=0)

    def _M_sig(self, sig):
        '''Calculate M(\sigma)'''
        r0 = self.cosmo.sigma(sig, z=0, inverse=True)
        return 4 / 3 * math.pi * r0**3 * self.rho

    def _deriv_S(self, M):
        '''Calculate dS/dM'''
        r0 = self._R(M)
        fac2 = 1 / 3
        sg_d = self.cosmo.sigma(r0, z=0, derivative=True)
        return sg_d * fac2

    def _deltac(self, z):
        '''Calculate d\delta_c/dz'''
        a0 = self.cosmo.growthFactor(z)
        deltacn = 1.686 * self.cosmo.Om(z)**(0.0055)
        return deltacn / a0

    def _R(self, M):
        return (3 * M / (4 * math.pi * self.rho))**(1 / 3)

    def _omega(self, M, z):
        fac1 = self._deltac(z) / self._sigma(M)
        fac2 = 10**(-self._deriv_S(M))
        return fac1 * fac2

    def _delc_deriv(self, z):
        dz = 0.001
        return (self._deltac(z + dz) - self._deltac(z)) / dz

    def _sigM_deriv(self, M):
        sig = self._sigma(M)
        r0 = self._R(M)
        fac2 = r0**(-2) / (4 * math.pi * self.rho)
        sg_d = self.cosmo.sigma(r0, z=0, derivative=True)
        return sg_d * (sig / r0) * fac2

    def _prob1(self, M0, z0):
        fac1 = 1 / (1 + (self._omega(M0, z0) / 4)**6)
        fac2 = self._omega(M0, z0)
        return fac1 * fac2

    def _prob0(self, z, M0, z0):
        delta0 = self._deltac(z0)
        sigma0 = self._sigma(M0)
        comp = 1 - (np.log10(self._deltac(z)) - np.log10(delta0)) / (0.272 / self._omega(M0, z0))
        ans1 = self._prob1(M0, z0) * max(0, comp)
        return ans1

    def get_accf_nop(self, Mn, zf, z0, M0):
        '''
            Get the accretion rate of dark matter halos. Neglecting the second term of Equation 10 in 
            Zhao et al.(2008).
        '''
        dervn = self._omega(Mn, zf)
        return dervn / 5.85

    def get_accf(self, Mn, zf, z0, M0):
        '''
            Get the accretion rate of dark matter halos
        '''
        dervn = self._omega(Mn, zf)
        return (dervn - self._prob0(zf, M0, z0)) / 5.85

    def get_growth(self, Mh, zlist):
        '''
        Functions used to calculate the mass assembly history of given halos.

        Input:
        ---------
        Mh: float
        Halo mass   unit:10^10 Msun

        zlist: float array
        An array of redshifts in ascending order. 
    
        Output:
        ---------
        Mlist: float array
        An array of halo masses for given halo at redshits specified by zlist
        '''
        M = Mh
        z_start = zlist[0]
        sigmalist = []
        Mlist = []
        delc_list = [self._deltac(z1) for z1 in zlist]

        zf = z_start
        delcf = delc_list[0]
        Mn = float(Mh)
        sigma_i = self._sigma(Mn)

        Mlist.append(Mn)
        sigmalist.append(sigma_i)

        for zc, delc in zip(zlist[1:], delc_list[1:]):
            dlndc = np.log(delc) - np.log(delcf)
            accf = self.get_accf(Mn, zf, z_start, Mh)
            lns_n = np.log(sigma_i) + accf * dlndc
            Mn = self._M_sig(np.exp(lns_n))

            zf = zc
            delcf = delc
            sigma_i = np.exp(lns_n)
            Mlist.append(Mn)
            sigmalist.append(sigma_i)

        sigmalist = np.array(sigmalist)
        Mlist = np.array(Mlist)
        return Mlist
    
    def get_growth_rate(self, Mn, zf, z0, Mh):
        '''
            Calculating the growth rate of the halo mass at redshift zf and mass Mn. The halo are found to have its mass equal to Mh at redshifts z0.(z0<zf)

            Input:
            -------------
                Mn: float array
                    The halo mass observed at redshifts specificed by zf. It can be quickly acquired by running get_growth().

                zf: float array
                    The redshifts to specify the mass of the halo in its assembly history.

                z0: float
                    The final redshifts of the halo

                Mh: float
                    The mass of the host halo at the final time.

            Output:
            -------------
                acc_rate: float array
                    The mass accretion rate of the host halo at redshifts specified by zn.
        '''
        accf = self.get_accf_nop(Mn, zf, z0, Mh)
        fac1 = self._sigM_deriv(Mn)
        fac2 = self._delc_deriv(zf)
        fac3 = self._sigma(Mn)/self._deltac(zf)
        acc_rate = accf/fac1*fac2*fac3
        return acc_rate