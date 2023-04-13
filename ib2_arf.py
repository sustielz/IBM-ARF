import numpy as np
from time import time
from ibmac2 import IBMAC2
from ib2_amr import IB2_AMR, curvature
class IB2(IBMAC2, IB2_AMR): pass
from acoustic_forces_vec_simple import *
global k
lamb = 750   #um
k = 2*np.pi/lamb     #wavenumber

class IB2_ARF(IB2):
    def __init__(self, fluid, *args, **kwargs):
#         print(*args)
        super(IB2_ARF, self).__init__(*args, **kwargs)
        self.Force = lambda X: self._Force(X, fluid.t)
        self.L=fluid.L
        
    def _Force(self, X, t):
        self._Fs = self.K*curvature(X)/self.dtheta               #### ST is along inward normal
        rr, RR, nhat = to_sph2(X)
        a = get_a_spw(np.mean(X[:, 1])-self.L/2)
        F = acf(rr, RR, nhat, a, self.Tamp)
        # fn = dot(F, nhat); F -= (fn-np.mean(fn))[..., np.newaxis]*nhat
        self._Fac=F/k**2               # Rescale force so Tamp=300 is appropriate for Nb=100        
        return self._Fs + self._Fac
    