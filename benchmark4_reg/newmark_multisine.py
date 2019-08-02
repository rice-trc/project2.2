#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from pyvib.nonlinear_elements_newmark import NLS, Tanhdryfriction
from pyvib.newmark import Newmark
from pyvib.forcing import multisine
from pyvib.common import dsample


# load system defined in matlab
data = loadmat('data/system.mat')
M = data['M']
C = data['C']
K = data['K']
muN = data['muN'].item()
eps = data['eps_reg'].item()
T_tip = data['T_tip'].squeeze().astype(int)
Fex1 = data['Fex1'].squeeze().astype(int)
w = data['w'].squeeze().astype(int)
ndof = M.shape[0]
# Fixed contact and free natural frequencies (rad/s). 
om_fixed = data['om_fixed'].squeeze()
om_free = data['om_free'].squeeze()

# define multisine
f0 = 10
f1 = 100
Nt = 2**13
R = 2
P = 6
fs = 750
A = 10
ns = Nt*R*P
t = np.arange(ns)/fs

upsamp = 20
fsint = upsamp * fs
Ntint = upsamp * Nt
# add extra period whci will be removed due to edge effects
Pfilter = 1
if upsamp > 1:
    P = Pfilter + P
nsint = Ntint*P*R
dt = 1/fsint

u, lines, freq  = multisine(f0,f1,N=Ntint,fs=fsint,R=R,P=P)
fext = np.zeros((nsint, ndof))
fext[:,Fex1==1] = A*u.ravel()[:,None]

nls = NLS(Tanhdryfriction(eps=eps, w=w, kt=muN))
sys = Newmark(M,C,K,nls)
x, xd, xdd = sys.integrate(fext,dt,x0=None, v0=None, sensitivity = False)

# We need to reshape into (npp,m,R,P)
fext = fext.reshape(R,P,Ntint,ndof).transpose(2,3,0,1)
# fext.shape = (Ntint,P,R,ndof)
x = x.reshape(R,P,Ntint,ndof).transpose(2,3,0,1)
xd = xd.reshape(R,P,Ntint,ndof).transpose(2,3,0,1)
xdd = xdd.reshape(R,P,Ntint,ndof).transpose(2,3,0,1)

try:
    if upsamp:# > 1:
        x2 = dsample(x, upsamp)
        xd2 = dsample(xd, upsamp)
        xdd2 = dsample(xdd, upsamp)
        fext2 = fext[::upsamp]
except:
    x2, xd2, xdd2, fext2 = [None]*4

np.savez('data/ms.npz',x=x,xd=xd,xdd=xdd, x2=x2,xd2=xd2,xdd2=xdd2,
         fext=fext, fext2=fext2)
#plt.figure()
#plt.plot(t, x, '-k', label=r'$x_1$')
##plt.plot(t, x, '-r', label=r'$x_2$')
#plt.xlabel('Time (t)')
#plt.ylabel('Displacement (m)')
#plt.title('Force type: {}, periods:{:d}')
#plt.legend()
#
#plt.figure()
#plt.plot(np.abs(np.fft.fft(x[6*1024:7*1024,0])))
#
#
#plt.show()
