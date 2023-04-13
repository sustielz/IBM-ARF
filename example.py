#### Script for IBM simulation of a single droplet in incompressible, periodic fluid

### General Setup
import numpy as np 
import warnings; warnings.simplefilter('ignore')
import sys
sys.path.append('src')
import os
import json

from util import CIRCLE, SUNFLOWER          #### General functions (iterate, geometry, force functions, etc)

from fluidmac2 import FLUIDMAC as FLUID
from ib2_arf import IB2_ARF
from ibmac2 import IBMAC2
from pib2 import PIB2 as PIB20
class PIB2(IBMAC2, PIB20): pass             


from time import time

#### Iterate fluids and immersed solids using built-in functions
def iterate(fluid, solids):
    ff = 0. 
    ## Force density on fluid
    for solid in solids:
        solid.step_XX(fluid.u)
        ff += solid.ff         # Force at midpoint
#     ff += fluid.ff             # External force on fluid. NOTE: add this after solid.ff to keep numpy happy
    uu=fluid.step_u(ff)        # Step Fluid Velocity
    for solid in solids:
        solid.step_X(uu)       # full step using midpoint velocity    

########    I/O    ########
#### Load Default Parameters into Variables ####
#### argv are for custom parameters.
#### IN BASH: run as python3 script.py K 100 dt 0.005 ... [paramname] [paramval]
ARGV = sys.argv


with open('default_params.json', 'r') as f:
    params = json.load(f)
   
locals().update(params)
############################


k = 2*np.pi/lamb     #wavenumber 
omega = 2*np.pi/(Tper*dt)
rad *= L
pos = np.array([x0, y0])*L
node*=L
# node += 2.27*100/Tamp  #### Preliminary test shows dz~2.27 for Tamp=100

# tilt = np.pi*stilt    #### Tilt not currently implemented for Legendre basis; require spherical harmonics / associated legendre polynomials
# psi=0.

def proj(X, p): return np.sum(X*p, axis=1)[:, np.newaxis]*p
############ Subclass IB2 to add acoustic forces at exterior     ##########


# ####################################
#   ########   Simulation   ########
# ####################################

# #### Initialize Fluid+Droplets
fluid = FLUID(N=N, L=L, mu=mu, dt=dt)
inside = PIB2(SUNFLOWER(rad-fluid.h/2, pos, n=Ni), fluid.N, fluid.h, fluid.dt)
inside.Kp = Kp    
inside.M = M or inside.M
inside.g=g
# outside = IB2(CIRCLE(rad, pos, Nb), fluid.N, fluid.h, fluid.dt, K=K/100)
outside = IB2_ARF(fluid, CIRCLE(rad, pos, Nb), fluid.N, fluid.h, fluid.dt, K=K/100) #### Rescale K so that K=1->K=.01 is reasonable for Nb=100
 
outside.n_tol = sn_tol         #### Tolerance for surface tension spacing refinement
outside.n_max *= sn_max    
outside.a *= sa

outside.L=L
outside.node=node
outside.Tamp=np.sqrt(Tamp)
outside.tilt=tilt

NRUN = 0                        ## Is there data from an existing simulation that we want to continue?
while os.path.exists('data/run{}.npz'.format(NRUN)):
   NRUN+=1
if NRUN>0:                      ## Load most recent data from last state
    data = np.load('data/run{}.npz'.format(FILENAME, NRUN-1))
    fluid.u = data['U'][-1]
    fluid.pp2 =    data['P'][-1]
    outside.X = data['Xout'][-1]
    outside.X0 = data['Xout0'][-1]
    inside.X =   data['Xin'][-1]
    inside.Y =     data['Y'][-1]
    del data

solids = [outside, inside]

#### Values that we're tracking
delta = []        #### record delta every step
data1 = [[] for i in range(7)]  #### Record droplet info every iteration (nmod steps)
data2  = [[], []]   #### Record fluid fields every 100 iterations (100*nmod steps)
TEMP=[[] for i in range(7)]             #### Container to let us update fluid+droplet data  simultaneously

t0 = time()
for i in range(nsteps+1):
    iterate(fluid, solids)
    delta.append(np.max(np.linalg.norm(inside.Y - inside.X, axis=1)))
#    aart.append(outside.aart.copy())
    if i%nmod==0:
        print('{}: time = {:.3e} ms || runtime (min) = {:.3e}'.format(i, dt*i, (time()-t0)/60))
        # print('{}: time = {:.3e} ms || runtime (min): TOT = {:.3e} | ST = {:.3e} | REF = {:.3e} | AC = {:.3e} | F = {:.3e}'.format(i, dt*i, (time()-t0)/60, solids[0].calctimeS/60, solids[0].calctimeR/60, solids[0].calctimeA/60, solids[0].calctimeF/60  ))

        temp=[outside.X0.copy(), outside.X.copy(), inside.X.copy(), inside.Y.copy(), 0, outside._Fs.copy(), outside._Fac.copy()]
        for j, val in enumerate(temp): TEMP[j].append(val)
        if i%(nmodu*nmod)==0:
            data2[1].append(fluid.pp2.copy())
            data2[0].append(fluid.u.copy())
            for j, val in enumerate(TEMP): data1[j].extend(val)
            TEMP = [[] for j in range(7)]
print(time()-t0)
with open('data/run{}.npz'.format(FILENAME, NRUN), 'wb') as f:
    np.savez(f,
        delta=np.array(delta),
        U=np.array(data2[0]),
        P=np.array(data2[1]),
        Xout0=np.array(data1[0]),
        Xout=np.array(data1[1]),
        Xin=np.array(data1[2]),
        Y=np.array(data1[3]),
        aart=np.array(data1[4]),
        Fs=np.array(data1[5]),
        Fac=np.array(data1[6]))

