### General Setup
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from matplotlib import patches
import json
import sys
import os
from time import time

## rotate counterclockwise
def rot(X, phi): return np.matmul(X, np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]))
def proj(X, p): return np.sum(X*p, axis=1)[:, np.newaxis]*p

#################### I/O ####################


with open('default_params.json', 'r') as f:
    params = json.load(f)
locals().update(params)
IM1NAME = 'ims/im1'
IM2NAME = 'ims/im2'

DATANAME = 'datac/run.npz'
DATA = np.load(DATANAME)
locals().update(DATA)
#############################################


freq = 2e3
omega = 2*np.pi*freq
def vorticity(u):
    N = np.shape(u)[1]
    ii = np.arange(N)
    ip = (ii+1)%N
    im = ii-1
    vorticity=(u[1][np.meshgrid(ip, ii)]
                   -u[1][np.meshgrid(im,ii)]
                   -u[0][np.meshgrid(ii,ip)]
                   +u[0][np.meshgrid(ii,im)])/(2*N/L)
    return vorticity


cmap = plt.get_cmap('tab10'); RED = plt.get_cmap('Reds')

ani_stats = plt.figure()

#### Figure for Stats #######
fig_stats = plt.figure(figsize=(10,10))
axs1 = fig_stats.add_subplot(321)
axs1.set_ylabel('|Y-X|/h'); axs1.set_xlabel('time'); axs1.set_title('Stability')

axs2 = fig_stats.add_subplot(322)
axs2.set_ylabel('Area'); axs2.set_xlabel('time'); axs2.set_title('Area vs Time')
axs3 = fig_stats.add_subplot(323)
axs3.set_ylabel('Axis Length'); axs3.set_xlabel('time'); axs3.set_title('Major/Minor Axes vs Time')

axs4 = fig_stats.add_subplot(324)
axs4.set_title('Center of Mass Trajectories')

axs5 = fig_stats.add_subplot(325)
axs5.set_ylabel('Max Vorticity'); axs5.set_xlabel('time'); axs5.set_title('Tilt vs Time')

axs6 = fig_stats.add_subplot(326)
axs6.set_title('Velocity of COM Trajectories')




t = dt*nmod*np.arange(len(Xin))
t0 = 0
X0 = Xout - np.mean(Xout, axis=1)[:, np.newaxis, :]
radii = np.linalg.norm(X0, axis=2)
# rmax = np.max(radii)
rmax = 1.25*rad*L
theta = np.arctan2(X0[:,:,0], X0[:,:,1])

nhat = Fs/np.linalg.norm(Fs, axis=2)[:,:,np.newaxis]
rhat = X0/radii[:,:,np.newaxis]
thhat = rot(rhat, -np.pi/2)  ## NOTE: This is NOT the theta hat used in calcs/spherical coords; here thhat always runs clockwise, used to determine net torques and stuff in 2D
nr = np.sum(nhat*rhat, axis=2)
nhat[nr<0] *= -1 # In computations of acoustic force, take always the outward normal
that = rot(nhat, -np.pi/2)   ## Note that that is just rotated-nhat, as is our convention

fn = np.sum(Fac*nhat, axis=2)
ft = np.sum(Fac*that, axis=2)

axs1.plot(t0+t, [np.max(np.linalg.norm(Y[i]-Xin[i], axis=1))*N/L for i in range(len(Y))], color=cmap(0))

dtheta = theta - np.roll(theta, 1, axis=1)
dtheta[dtheta>=np.pi] -= 2*np.pi
dtheta[dtheta<=-np.pi] += 2*np.pi
axs2.plot(t0+t, np.sum(dtheta*radii**2, axis=1)/(2*np.pi), color=cmap(0))

arclen = np.linalg.norm(X0 - np.roll(X0, -int(np.shape(X0)[1]/2), axis=1), axis=2)  #### Length of axis
a = np.max(arclen, axis=1)
b = np.min(arclen, axis=1)
axs3.plot(t0+t, a, label='a', color=cmap(0)); axs3.plot(t0+t, b, label='b', color=cmap(1))

COMY = np.mean(Y, axis=1)
COMin = np.mean(Xin, axis=1)
COMout = np.mean(Xout, axis=1)
axs4.plot(COMY[:,0], COMY[:, 1], label='COM Y', color=cmap(0), marker='.'); axs4.plot(COMin[:,0], COMin[:, 1], label='COM Xin', color=cmap(1), marker='.'); axs4.plot(COMout[:,0], COMout[:, 1], label='COM Xout', color=cmap(2), marker='.')



VCOM = np.diff(COMout, axis=0)/(dt*nmod)
VCOMi = np.diff(COMin, axis=0)/(dt*nmod)

axs6.plot(VCOM[:,0], VCOM[:, 1], label='VOM Xout', color=cmap(0), marker='.'); axs5.plot(VCOMi[:,0], VCOMi[:, 1], label='VOM Xin', color=cmap(1), marker='.')


t = dt*nmod*nmodu*np.arange(len(U))

_t0 = time()
print('calculating vorticity...')
w = np.stack([vorticity(u) for u in U], axis=0)
print('found vorticuty in {}'.format(time()-t0))
wmin = np.min(np.min(w, axis=-1), axis=-1)
wmax = np.max(np.max(w, axis=-1), axis=-1)

axs5.plot(t, wmax)
wmin=np.min(wmin)
wmax = np.max(wmax)

h, l = axs3.get_legend_handles_labels(); axs3.legend(h[-2:], l[-2:])
h, l = axs4.get_legend_handles_labels(); axs4.legend(h[-3:], l[-3:])
h, l = axs6.get_legend_handles_labels(); axs6.legend(h[-3:], l[-3:])

fig_stats.tight_layout(pad=0.25)
fig_stats.savefig(IM1NAME+'.png')

#### Animation
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(4,4)
ax = [fig.add_subplot(gs[ :2, :2]),
      fig.add_subplot(gs[ :2,2:4]),
      fig.add_subplot(gs[2,:2]),
      fig.add_subplot(gs[3,:2]),
      fig.add_subplot(gs[2:4, 2:4])]

ims = []

ax[0].set_xlim([0, L]); ax[0].set_ylim([0, L])

ax[1].set_title('$K={}, M={}, N_i={}$'.format(K, M, Nb))
ax[1].set_aspect('equal', 'box')
#     ax2.set_ylim([-1, 1])

ax[1].set_xlim([-1.1*rmax, 1.1*rmax])
ax[1].set_ylim([-1.1*rmax, 1.1*rmax])



N_steady_state = int(200/nmodu)
umin = np.min(np.linalg.norm(U, axis=1)[N_steady_state:]) - np.min(np.linalg.norm(VCOM, axis=1)[N_steady_state:]) - 1e-6
umax = np.max(np.linalg.norm(U, axis=1)[N_steady_state:]) + np.max(np.linalg.norm(VCOM, axis=1)[N_steady_state:]) + 1e-6
for ii, u in enumerate(U[:]):
    print(ii)
    i=ii*nmodu
    if i>= len(Xout): continue
    im=[]
    im.append(ax[0].imshow(w[ii], vmin=wmin, vmax=wmax, origin='lower', extent=[0,L,0,L]))
    if i==0: fig.colorbar(im[-1], ax=ax[0])
    im.append(ax[0].text(0.5, 1.01, 'Vorticity/Pos at Time {} (ms)'.format(i*nmod*dt), horizontalalignment='center', verticalalignment='bottom', transform=ax[0].transAxes))
    im.append(ax[0].scatter(Xout[i][:,0]%L, Xout[i][:,1]%L, s=100/Ni, color=cmap(1)))
#    im.append(ax[0].quiver(*Xout[i][::4].T, *Fac[i][::4].T, color='red', width=.004))

    #### Plot Droplet in COM Frame
    com = COMout[i]
    try:
        vcom = VCOM[i]
    except:
        pass
#    print(com)
    ins = Xin[i] - com
    out = Xout[i] - com

    try:
        dN = int(rmax*1.1*N/L)
        dNRange = np.arange(-dN, dN+1)
        Ncom = [int(c*N/L) for c in com]
        COORDS = np.meshgrid(Ncom[0]+dNRange, Ncom[1]+dNRange)
        GRID=[(COORDS[j]-Ncom[j])*L/N for j in range(2)]
        UGRID=[u[j][COORDS]-vcom[j] for j in range(2)]

        # SPL = ax[1].streamplot(*GRID, *UGRID, color=np.sqrt(UGRID[0]**2+UGRID[1]**2)/(umax-umin), cmap=plt.get_cmap('hot'))
        SPL = ax[1].streamplot(*GRID, *UGRID, color=np.sqrt(UGRID[0]**2+UGRID[1]**2), norm=plt.Normalize(umin, umax), cmap=plt.get_cmap('hot'))
        im.append(SPL.lines)
        if i==0: fig.colorbar(im[-1], ax=ax[1])

        ax[1].patches = []
    except:
        print('failed to draw streamlines!')
        pass
    im.append(ax[1].scatter(ins[:,0], ins[:,1], s=1000/Ni, color=cmap(0)))
    im.append(ax[1].scatter(out[:,0], out[:,1], s=1000/Ni, color=cmap(1), zorder=0))
    im.append(ax[1].scatter([out[0,0]], [out[0,1]], color='red'))  ## Mark theta=0
    im.append(ax[1].axhline(y=.5*L-com[1], color=RED(0.1+0.45*(1+np.sin(2*np.pi*i*nmod*freq)))))

    yins = Y[i] - com
    im.append(ax[1].scatter(yins[:,0], yins[:,1], s=1000/Ni, color=cmap(6)))


    im.append(ax[4].text(0.5, 1.01, 'mean u: {:.3e}, {:.3e}'.format(np.mean(u[0]), np.mean(u[1])), horizontalalignment='center', verticalalignment='bottom', transform=ax[4].transAxes))

    im.append(ax[4].imshow(np.linalg.norm(u, axis=0).T, vmin=umin, vmax=umax, origin='lower', extent=[0,L,0,L], cmap=plt.get_cmap('hot')))
    # if i==0: fig.colorbar(im[-1], ax=ax[4])
    im.extend(ax[4].plot(Xout[i][:,0]%L, Xout[i][:,1]%L, color='white'))
    try:
        SPL = ax[4].streamplot(*np.meshgrid(np.arange(N)*L/N, np.arange(N)*L/N), u[0].T, u[1].T, color=w[i]/(wmax-wmin), density=2.5)
        im.append(SPL.lines)
        ax[4].patches = []
    except:
        print('failed to draw streamlines! (in second plot)')
        pass



    im.extend(ax[2].plot(theta[i], radii[i], color='black'))
    #im.extend(ax[2].plot(np.linspace(-np.pi, np.pi, len(radii[i])), radii[i], color='black', linestyle='--'))
    im.append(ax[2].axvline(x=np.pi/2, color='black', linestyle='--'))
    im.append(ax[2].axvline(x=-np.pi/2, color='black', linestyle='--'))
    im.append(ax[2].text(0.5, 1.01, 'dPeaks: {:.3e}'.format(np.max(radii[i][theta[i]>0]) - np.max(radii[i][theta[i]<0])), horizontalalignment='center', verticalalignment='bottom', transform=ax[2].transAxes))
    im.extend(ax[3].plot(theta[i], fn[i], color='black'))
    im.extend(ax[3].plot(theta[i], ft[i], color='red'))
#    im.extend(ax[3].plot(-theta[i], fn[i], color='black', linestyle='--'))
#    im.extend(ax[3].plot(-theta[i], ft[i], color='red', linestyle='--'))
    im.append(ax[3].axvline(x=0, color='black', linestyle='--'))
    im.append(ax[3].axhline(y=0, color='black', linestyle='--'))
    im.append(ax[3].text(0.5, 1.01, 'Sum ft[i]: {:.3e}'.format(sum(ft[i])), horizontalalignment='center', verticalalignment='bottom', transform=ax[3].transAxes))
    ims.append(im)

fig.tight_layout()

# axs5.axhline(y=stilt, color='black', linestyle=':', label='Trapping Plane Tilt')






        #### DEBUG ####
#         yins = Y[j][i] - com
#         im.append(axj.scatter(yins[:,0], yins[:,1], s=1000/Ni, color=cmap(1)))
#         TETHERS = np.array([[ins[i], yins[i]] for i in range(len(ins))])
#         for tether in TETHERS:
#             im.extend(axj.plot(tether[:,0], tether[:,1], color=RED(np.linalg.norm(tether[1]-tether[0])*N/L)))

        ## Record theta profile of each boundary
#             THETA[j].append(np.arctan2(out[:,1], out[:,0]))



print(len(ims))
ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1)
ani.save(IM2NAME+'.gif', writer='pillow')

                                                                                                        