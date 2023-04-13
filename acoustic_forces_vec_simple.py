import numpy as np
global k
lamb = 750              #levitator size (um)
k = 2*np.pi/lamb        #wavenumber

def rot(X, phi): return np.matmul(X, np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])) # rotate counterclockwise
def rot90(X): return np.stack([X[..., 1], -X[..., 0]], axis=-1)
def dot(a, b): return np.sum(a*b,axis=1)



##################################################################
  #####################   Geometry   ###########################
##################################################################
def center(X, x0=None): return X - np.mean(X, axis=0) if x0 is None else X - x0
def to_rth(X0): return np.array([np.linalg.norm(X0, axis=1), np.arctan2(X0[:,0], X0[:,1])])
def to_xy(rth):  return np.stack([rth[0]*np.sin(rth[1]), rth[0]*np.cos(rth[1])], axis=1)
def to_rthhat(X0, rth):   
    rhat = X0/rth[0, :, np.newaxis]
    # thhat = rot(rhat, -np.pi/2)
    thhat = rot90(rhat)
    thhat[rth[1]<0]*=-1                   ## By our convention, thhat flips (to point downwards) for th<0 
    return [rhat, thhat]    
    
def get_n(X):
    dX = np.diff(X, axis=0, prepend=X[-1].reshape([1, 2]))
    dX/= np.linalg.norm(dX, axis=1)[:, np.newaxis]
    return np.diff(dX, axis=0, append=dX[0].reshape([1, 2]))

def get_nhat(X, rhat=None):             ## Pass rhat to flip any inward-facing normals
    n = get_n(X)
    nhat = n/np.linalg.norm(n, axis=1)[:, np.newaxis]
    if rhat is not None: nhat[dot(nhat, rhat)<0] *= -1  ## By our convention, nhat points outwards 
    return nhat

def to_sph2(X, x0=None):
    X0 = center(X, x0)
    rth = to_rth(X0)
    rthhat = to_rthhat(X0, rth)
    nhat = get_nhat(X, rthhat[0])
    return rth, rthhat, nhat


                    
##################################################################
  ######################   Fields   ############################
##################################################################

        ###########   Spherical Functions   ###########  
from scipy.special import spherical_jn, spherical_yn
def jl(l, z, derivative=False): return spherical_jn(l, z, derivative)
def yl(l, z, derivative=False): return spherical_yn(l, z, derivative)
def hl(l, z, derivative=False): return jl(l,z,derivative)+1j*yl(l,z,derivative)

from scipy.special import legendre
def Pl(j, th): return legendre(j)(np.cos(th)) if j>=0 else 0*th
def dPl(j, th): return -legendre(j).deriv()(np.cos(th))*np.sin(abs(th)) if j>=1 else 0*th

    ###########   Field @ position rth, mode lm   ###########  
def phl(rth, l): return jl(l, k*rth[0])*Pl(l, *rth[1:])
def url(rth, l): return k*jl(l, k*rth[0], True)*Pl(l, *rth[1:])
def uthl(rth, l): return jl(l, k*rth[0])*dPl(l, *rth[1:])/rth[0]

def qhl(rth, l): return hl(l, k*rth[0])*Pl(l, *rth[1:])
def vrl(rth, l): return k*hl(l, k*rth[0], True)*Pl(l, *rth[1:])
def vthl(rth, l): return hl(l, k*rth[0])*dPl(l, *rth[1:])/rth[0]
  
          ###########   Scattering   ###########  
### Sol for the BC's where (1) rho*phi and (2) vn are both continuous inside/ouside spherical drop; e.g. for mode l
###    phi_ins == b j(kR) P(cos) = a(1+b2) j(kR) P(cos) 
###    phi_out == a [j(kR) + s h(kR)]P(cos) 
def scat_sph(R, rho_ex=.18, lmax=3): ## scattered wave
    rat=1/(1+rho_ex)
    S  = [(1-rat)/( rat*hl(l, k*R)/jl(l, k*R) - hl(l, k*R, True)/jl(l, k*R, True)  )  for l in range(lmax)]
    B2 = [S[l]*hl(l, k*R, True)/jl(l, k*R, True)                                      for l in range(lmax)]
    return np.array([S, B2])

## standing plane wave
# def get_a_spw(r0, lmax=3): return np.array([4*np.pi*np.cos(k*r0 + l*np.pi/2)*Pl(l, 0) for l in range(lmax)]) 
def get_a_spw(r0, lmax=3): return np.array([(2*l+1)*np.cos(k*r0 + l*np.pi/2)*Pl(l, 0) for l in range(lmax)]) 

## traveling plane wave  
# def get_a_pw(r0, lmax=3): return np.array([4*np.pi*(1j)**l*np.exp(1j*k*r0)*Pl(l, 0) for l in range(lmax)])
def get_a_pw(r0, lmax=3): return np.array([(2*l+1)*(1j)**l*np.exp(1j*k*r0)*Pl(l, 0) for l in range(lmax)]) 

        ###########   Total Fields   ###########  
def _collect(rr, c, FIELD): return sum([c[l]*FIELD(rr, l) for l in range(len(c))])
def collect(rr, c, FIELDS=[]): return [_collect(rr, c, FIELD) for FIELD in FIELDS]
def get_fields(rr, a, R, rho_ex=.18):
    S, B2 =  scat_sph(R, rho_ex=rho_ex, lmax=len(a))
    inc = collect(rr, a, FIELDS=[phl, url, uthl])
    tr = collect(rr, a*B2, FIELDS=[phl, url, uthl])
    sc = collect(rr, a*S, FIELDS=[qhl, vrl, vthl])
    return inc, tr, sc





####### Forces  ############
def correct_vel(vr, vth, nr, nth, rat, N=2):
        vn = nr*vr+nth*vth
        vt = nr*vth-nth*vr
        vr1 =  -nth*vt*(1/rat-1)
        vth1 = -nth*vn*(1/rat-1)
        if N==-1:
            return 0, 0
        if N==0:
            return vr1, vth1
        else:
            vr2, vth2 = correct_vel(vr1, vth1, nr, nth, rat, N-1)
            return vr1+vr2, vth1+vth2

def correct_fields(fld, nr, nth, rat, N=2):
    vrc, vthc = correct_vel(fld[1], fld[2], nr, nth, rat, N=N)
    return [1/rat*fld[0], fld[1]+vrc, 1/rat*fld[2]+vthc]

def acf(rr, RR, nhat, a, Tamp=10, rho_ex=.18, R=None, wt=None, no_inc2=True, no_sc2=False, N_cor=2):
    R=R or np.max(rr[0])
    rat = 1/(1+rho_ex)
    nr = dot(nhat, RR[0])
    nth = dot(nhat, RR[1])
    vec = lambda vr, vth: vr[:, np.newaxis]*RR[0] + vth[:, np.newaxis]*RR[1]
    
    inc, tr, sc = get_fields(rr, a,  R, rho_ex=rho_ex)
    inc=[Tamp*f for f in inc]; tr=[Tamp*f for f in tr]
    tot = [inc[i] + tr[i] for i in range(3)]
    totc = correct_fields(tot, nr, nth, rat, N=N_cor)
    
    ### outside incident-incident part / scattered-scattered part
    Fii = acoustic_force(inc[0], vec(inc[1], inc[2]), nhat, wt=wt)
    Fss = acoustic_force(tr[0], vec(tr[1], tr[2]), nhat, wt=wt)
    F = acoustic_force(totc[0], vec(totc[1], totc[2]), nhat, wt=wt)
   
    if no_inc2: F-= Fii
    if no_sc2: F -= Fss
    return F

def acoustic_force(ph, v, nhat, wt=None):
    rr = lambda f, g: np.real(f)*np.real(g)
    ii = lambda f, g: np.imag(f)*np.imag(g)
    ri = lambda f, g: np.real(f)*np.imag(g)
       
    if wt is None:
        sq = lambda f, g: (rr(f,g)+ii(f,g))/2
    else:
        sq = lambda f, g: rr(f,g)*np.cos(wt)**2 + ii(f,g)*np.sin(wt)**2 + (ri(f,g)+ri(g,f))*np.sin(2*wt)/2

    vn = np.sum(v*nhat, axis=1)[:, np.newaxis]*np.ones(np.shape(nhat))  ## reshaped for input to sq
    fn =  (k**2*sq(ph, ph) + np.sum(sq(v, v), axis=1))[:, np.newaxis]*nhat
    ft = -sq(vn, v) - sq(v, vn)
    # return -fn-ft if wt is None else -fn
    return (-fn-ft)/2
