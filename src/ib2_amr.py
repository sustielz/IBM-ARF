import numpy as np 
import matplotlib.pyplot as plt
from time import time

from ib2 import IB2         

#### Class for a 2D immersed boundary with surface tension in fluid. In addition to replacing the force functional, 
#### surface tension also requires even spacing of boundary points. This is accomplished here using 'virtual springs', which
#### do not exert force on the fluid but guide tangential motion along the boundary to adjust spacing. 

class IB2_AMR(IB2):
    
    def __init__(self, X, N, h, dt, **kwargs):
        super(IB2_AMR, self).__init__(X, N, h, dt, **kwargs)
        self.a = self.dt/6.         ## stiffness of vitrual springs. Note there is a stability restraint a ~< dt/4
        self.n_max = 50000
        self.n_tol = 0.1

        self.Force = lambda X: self.K*curvature(X)/self.dtheta
        self.calctimeR = 0
        self.REF = True
        self.CLEAN = False
        
        
        
    def step_X(self, uu): 
        super(IB2_AMR, self).step_X(uu)
        K, kp, km = self.K, self.kp, self.km
        self.X0 = self.X.copy()
        t0 = time()
        if self.REF: self.X, self.n_ref = refine(self.X, self.a, self.n_tol*self.h, self.n_max)
        self.calctimeR += time()-t0
        self.X1 = self.X.copy()
        if self.CLEAN: self.X = clean(self.X)


## rotate clockwise
def rot(X, phi): return np.matmul(X, np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]))

def curvature(X):  ## Surface tension
    dX = np.diff(X, axis=0, prepend=X[-1].reshape([1, 2]))
    dX/= np.linalg.norm(dX, axis=1)[:, np.newaxis]
    return np.diff(dX, axis=0, append=dX[0].reshape([1, 2]))

def virtual_spring(X, a):
    dX = np.diff(X, axis=0, prepend=X[-1].reshape([1, 2]))
    lX = np.linalg.norm(dX, axis=1)
    n_tol = max(lX) - min(lX)

    n = np.diff(dX/lX[:, np.newaxis], axis=0, append=dX[0].reshape([1, 2]))        
    un = n/np.linalg.norm(n, axis=1)[:, np.newaxis]
    ut = rot(un, -np.pi/2)

    Ft = a*np.diff(dX, axis=0, append=dX[0].reshape([1, 2]))      
    X += np.sum(ut*Ft, axis=1)[:, np.newaxis]*ut
    return n_tol


    
def refine(X, a, tol, nmax=500):
    dtheta=2*np.pi/len(X)
    a/=dtheta**2
    n_ref = 0
    n_tol = tol+1
    while n_tol > tol and n_ref < nmax:   ## Adjust tangential position until tolerance is reached
        n_tol = virtual_spring(X, a)
        n_ref += 1
    return X, n_ref  



################ Geometry ############
def to_rth(X0): return np.array([np.linalg.norm(X0, axis=1), np.arctan2(X0[:,0], X0[:,1])])
def to_xy(rr):  return np.stack([rr[0]*np.sin(rr[1]), rr[0]*np.cos(rr[1])], axis=1)

def clean(X, nstep=10):
    x0 = np.mean(X, axis=0) 
    rth = to_rth(X-x0)
    wr = np.fft.rfft(rth[0])
    wr[-nstep:]*=0
    rth[0]=np.fft.irfft(wr)
    return to_xy(rth)+x0
    
# import numpy as np 
# import matplotlib.pyplot as plt
# from time import time

# from ib2 import IB2         

# #### Class for a 2D immersed boundary with surface tension in fluid. In addition to replacing the force functional, 
# #### surface tension also requires even spacing of boundary points. This is accomplished here using 'virtual springs', which
# #### do not exert force on the fluid but guide tangential motion along the boundary to adjust spacing. 

# class IB2_AMR(IB2):
    
#     def __init__(self, X, N, h, dt, **kwargs):
#         super(IB2_AMR, self).__init__(X, N, h, dt, **kwargs)
#         self.a = self.dt/6.         ## stiffness of vitrual springs. Note there is a stability restraint a ~< dt/4
#         self.n_max = 50000
#         self.n_tol = 0.1

#         self.Force = lambda X: self.K*Force_surf(X)/self.dtheta
#         self.calctimeR = 0

# #     def Force_surf(self, X):  ## Surface tension
# #         K, kp, km = self.K, self.kp, self.km
# #         dX = X - X[km]
# #         lX = np.linalg.norm(dX, axis=1)
# #         return self.K*(dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis])/(self.dtheta)
             
#     def step_X(self, uu): 
#         super(IB2_AMR, self).step_X(uu)
#         K, kp, km = self.K, self.kp, self.km
#         n_ref = 0
#         n_tol = self.n_tol
#         self.X0 = self.X.copy()
#         t0 = time()
#         while n_tol > self.n_tol*self.h and n_ref < self.n_max:   ## Adjust tangential position until tolerance is reached
#             X = self.X
#             dX = X - X[km]
#             lX = np.linalg.norm(dX, axis=1)
#             n_tol = max(lX) - min(lX)
#             n_ref += 1

#             Fn = dX[kp]/lX[kp, np.newaxis] - dX/lX[:, np.newaxis]
#             un = Fn/np.linalg.norm(Fn, axis=1)[:, np.newaxis]
#             ut = np.matmul(un, np.array([[0, -1], [1, 0]]))
#             uFFt = self.Force_spring(X)/self.K
#             self.FFt = np.sum(ut*uFFt, axis=1)[:, np.newaxis]*ut
#     #         self.FFt[np.linalg.norm(self.FFt, axis=1) < 1e-1] = 0
#             self.X += self.a*self.FFt

#         self.calctimeR += time()-t0
#         self.n_ref = n_ref       #### DEBUG ####


# ## rotate clockwise
# def rot(X, phi): return np.matmul(X, np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]))

# def Force_surf(X):  ## Surface tension
#     dX = np.diff(X, axis=0, prepend=X[-1].reshape([1, 2]))
#     dX/= np.linalg.norm(dX, axis=1)[:, np.newaxis]
#     return np.diff(dX, axis=0, append=dX[0].reshape([1, 2]))
        
# def refine(X, a, tol, nmax=500):
#     dtheta=2*np.pi/len(X)
#     n_ref = 0
#     n_tol = tol+1
#     while n_tol > tol and n_ref < nmax:   ## Adjust tangential position until tolerance is reached
#         dX = np.diff(X, axis=0, prepend=X[-1].reshape([1, 2]))
#         lX = np.linalg.norm(dX, axis=1)
#         n_tol = max(lX) - min(lX)
#         n_ref += 1
        
#         n = np.diff(dX/lX[:, np.newaxis], axis=0, append=dX[0].reshape([1, 2]))        
#         un = n/np.linalg.norm(n, axis=1)[:, np.newaxis]
#         ut = rot(un, -np.pi/2)
        
#         Ft = np.diff(dX, axis=0, append=dX[0].reshape([1, 2]))/dtheta**2        
#         X += a*np.sum(ut*Ft, axis=1)[:, np.newaxis]*ut

# #         uFFt = self.Force_spring(X)/self.K
# #         self.FFt = np.sum(ut*uFFt, axis=1)[:, np.newaxis]*ut
# # #         self.FFt[np.linalg.norm(self.FFt, axis=1) < 1e-1] = 0

#     return n_ref       #### DEBUG ####

# from fluidmac2 import FLUIDMAC as FLUID
# from ib2_arf import IB2_ARF
# from ibmac2 import IBMAC2
# from pib2 import PIB2 as PIB20
# class PIB2(IBMAC2, PIB20): pass         


# from util import CIRCLE

# # for Nb in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
    