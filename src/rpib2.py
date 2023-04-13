import numpy as np 
import matplotlib.pyplot as plt

from ib2 import IB2


#### A penalty immersed boundary method for assigning mass density to a fluid. 

#### Massive fluid 'markers' consist of pairs of points connected by a stiff spring: 
#### a massless point X which moves at the local fluid velocity and applies force to the fluid; 
#### and a massive point Y which is subject to body forces but does not interact with the fluid. 
#### The force applied to the fluid at each point X is determined by the spring force to its respective mass point Y.


class RPIB2(IB2):
    
    @property
    def dtheta(self): return self.M/self.Nb
    
    def __init__(self, X, N, h, dt, K=1., M=0.01, Kp=5000):
        super(RPIB2, self).__init__(X, N, h, dt, K=K)
        self.Y = self.X.copy()      #### massive points Y initially coincide with fluid markers X
        self.M = M    
        self.Kp = Kp 
        self.beta = Kp*.01
        self.L = 0.                 #### In 2D, rotations are simplified since all rotations are around z-axis
        self.theta = 0.             ####     - L=Lz is a scalar
#         self.theta = 0.             ####     - orientation matrix E is rot(theta) of a scalar theta  
        
        self.YCM = np.mean(self.Y, axis=0)       
        self.VCM = np.zeros(2)
        self.C = self.Y - self.YCM[np.newaxis,:]        
        self.I0 = self.M*sum(np.linalg.norm(self.C, axis=1)**2)     #### Simplified since we only care about Lz
        self.I0i = 1./self.I0
        
        self.bForce = self.bForce_grav
    
    def step_XX(self, u): 
        super(RPIB2, self).step_XX(u)
        self.VV = 0.5*self.dt*(self.XX - self.X)
        self.YYCM = self.YCM+0.5*self.dt*self.VCM

        self.ttheta = self.theta - 0.5*self.dt*self.L/self.I0         #### TODO: Is this - sign correct?
        self.YY = self.YCM[np.newaxis, :] + rot(self.C, self.ttheta)  

#         self.EE = rot(self.E, -0.5*self.dt*self.L/self.I0)        

#         theta = -0.5*self.dt*self.L/self.I0      #### Simplified since L=Lz is not a vector
#         for i in range(2): self.EE[i] = rot(self.E[i], theta)            
#         self.YY = self.YCM[np.newaxis, :] + self.C.dot(self.EE)  #  self.EE.dot(self.C)   

        self.VVCM = self.VCM - 0.5*self.dt/self.M*np.mean(self.bForce(self.YY, self.YYCM) - self.FF, axis=0)    #### Factor of dtheta cancels from definition of M
        self.LL = self.L + 0.5*self.dt*(self.TT + self.bTorque(self.YY, self.YYCM))
#         self.LL = self.L - 0.5*self.dt*(self._bTorque - self.TT)
        return self.FF
        
    def step_X(self, uu):  # full step using midpoint velocity            
        super(RPIB2, self).step_X(uu)
        
        self.YCM += self.dt*self.VVCM  
        self.theta  -= self.dt*self.LL/self.I0   #### TODO: Is this - sign correct?
#         for i in range(2): self.E[i] = rot(self.E[i], theta)
#         self.E = rot(self.E, theta)
        
#         self.Y = self.YCM[np.newaxis, :] +  self.C.dot(self.E)  #self.E.dot(self.C)    
        self.Y = self.YCM[np.newaxis, :] +  rot(self.C, self.theta)    #self.E.dot(self.C)    
        self.VCM += self.dt/self.M*np.mean(self.bForce(self.YY, self.YYCM) - self.FF, axis=0)        
        self.L += self.dt*(self.TT + self.bTorque(self.YY, self.YYCM))
        
        return self.FF
       
    def Torque(self, Y, YCM, F):
        C = Y-YCM[np.newaxis, :]
#         return (self.dtheta/self.h**2)*np.sum(C[0]*F[1]-C[1]*F[0])
        return np.mean(C[:, 0]*F[:, 1]-C[:, 1]*F[:, 0])



    def bForce_grav(self, Y, YYCM):
        F = 0*Y
        F[:, 1] -= 9.8*self.M
        return F
    
        
#     def pForce(self, Y, X): return self.Kp*(Y-X)
    def pForce(self, Y, V, X): return self.Kp*(Y-X) - self.beta*V
    
    @property
#     def FF(self): return self.pForce(self.YY, self.XX) #+ self.Force(self.XX)
    def FF(self): return self.pForce(self.YY, self.VV, self.XX) #+ self.Force(self.XX)
    
    @property
    def ff(self): return self.vec_spread(self.FF, self.XX) # Force at midpoint
    
    def bTorque(self, Y, YCM): return self.Torque(Y, YCM, self.bForce(Y, YCM))

    
    @property
    def TT(self): return self.Torque(self.YY, self.YYCM, -self.FF) 


def rot(X, phi): return np.matmul(X, np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]))

    
    
    