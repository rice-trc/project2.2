#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from pyvib.common import db, dsample
from pyvib.forcing import multisine
from pyvib.newmark import Newmark
from pyvib.nonlinear_elements_newmark import NLS, Tanhdryfriction

"""This script simulates a cantilever beam with attached slider

The slider is attached to the end; In order to know the slider velocity, needed
for output-based identification, the slider is modeled as a small extra mass
attached to the tip with a spring. The nonlinear damping is then found from the
extra mass' velocity using a regulized tanh function, ie

fnl = μ*tanh(ẏ/ε)

To determine the right multisine amplitude, we make a with scan with increasing
amplitudes for one period and one realization. By looking at the first modes
resonance peak in the FRF, we can roughly correlate the amplitude to stick or
slip condition. We know the natural frequencies for each extreme from linear
modal analysis, ie. either fully stuck or fully sliding.

ωₙ = 19.59, 122.17, 143.11  # free
ωₙ = 21.44, 123.34, 344.78  # stuck
"""

scan = True
benchmark = 4

# define multisine
f0 = 5
f1 = 70
Nt = 2**13
fs = 15000

if scan:
    R = 1
    P = 1
    Avec = [10, 30, 50, 70, 80, 100, 120]
    Avec = [0.1,1,5,10,15,30,40,50,70,80,100,120,150]
    #Avec = [50]
    upsamp = 1
    fname = 'scan'
else:
    R = 2
    P = 6
    Avec = [20]
    upsamp = 20
    fname = 'ms'

ns = Nt*R*P
t = np.arange(ns)/fs
fsint = upsamp * fs
Ntint = upsamp * Nt
# add extra period which will be removed due to edge effects
Pfilter = 1
if upsamp > 1:
    P = Pfilter + P
nsint = Ntint*P*R
dt = 1/fsint


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
fdof = np.argwhere(Fex1).item()
nldof = np.argwhere(w).item()
ndof = M.shape[0]
# Fixed contact and free natural frequencies (rad/s).
om_fixed = data['om_fixed'].squeeze()
om_free = data['om_free'].squeeze()

#data2 = loadmat('data/b4_A1_up1_ms_full.mat')
#u = data2['u'].squeeze()
#freq = data2['freq'].squeeze()
#fs = data2['fs'].item()
#dt = 1/fs
#nsint = len(u)
#Ntint = nsint

np.random.seed(0)
u, lines, freq = multisine(f0, f1, N=Ntint, fs=fsint, R=R, P=P)
fext = np.zeros((nsint, ndof))

nls = NLS(Tanhdryfriction(eps=eps, w=w, kt=muN))
sys = Newmark(M, C, K, nls)
for A in Avec:
    fext[:, fdof] = A*u.ravel()
    print(f'Newmark started with ns: {nsint}, A: {A}')
    try:
        x, xd, xdd = sys.integrate(fext, dt, x0=None, v0=None,
                                   sensitivity=False)
        if scan:
            np.savez(f'data/scan_A{A}.npz', x=x, freq=freq, Ntint=Ntint,
                     fdof=fdof, nldof=nldof)
            # plot frf for forcing and tanh node
            Y = np.fft.fft(x[-Ntint:, [fdof, nldof]], axis=0)
            nfd = Y.shape[0]//2
            plt.figure()
            plt.plot(freq[:nfd], db(np.abs(Y[:nfd])))
            plt.xlim([0, 50])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.legend(['Force dof', 'nl dof'])
            plt.minorticks_on()
            plt.grid(which='both')
            plt.savefig(f'fig/nm_b{benchmark}_A{A}_fft_comp_n{fdof}.png')
    except ValueError as e:
        print(f'Newmark integration failed with error {e}. For A: {A}')

        # We need to reshape into (npp,m,R,P)
fext = fext.reshape(R, P, Ntint, ndof).transpose(2, 3, 0, 1)
# fext.shape = (Ntint,P,R,ndof)
x = x.reshape(R, P, Ntint, ndof).transpose(2, 3, 0, 1)
xd = xd.reshape(R, P, Ntint, ndof).transpose(2, 3, 0, 1)
xdd = xdd.reshape(R, P, Ntint, ndof).transpose(2, 3, 0, 1)

x2, xd2, xdd2, fext2 = [None]*4
if upsamp:  # > 1:
    x2 = dsample(x, upsamp)
    xd2 = dsample(xd, upsamp)
    xdd2 = dsample(xdd, upsamp)
    fext2 = fext[::upsamp, :, :, 1:]  # dsample remove first period

np.savez(f'data/{fname}.npz', x=x, xd=xd, xdd=xdd, x2=x2, xd2=xd2, xdd2=xdd2,
         fext=fext[:, fdof], fext2=fext2[:, fdof],
         lines=lines, fs=fs, A=A)
# plt.figure()
#plt.plot(t, x, '-k', label=r'$x_1$')
##plt.plot(t, x, '-r', label=r'$x_2$')
#plt.xlabel('Time (t)')
#plt.ylabel('Displacement (m)')
#plt.title('Force type: {}, periods:{:d}')
# plt.legend()
#
# plt.figure()
# plt.plot(np.abs(np.fft.fft(x[6*1024:7*1024,0])))
#
#
# plt.show()
