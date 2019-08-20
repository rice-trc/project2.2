#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat

from pyvib.common import db, dsample
from pyvib.forcing import multisine  # , multisine_time
from pyvib.modal import mkc2ss
from pyvib.newmark import Newmark
from pyvib.nlss import NLSS  # , dnlsim2
from pyvib.nonlinear_elements import NLS, Tanhdryfriction

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
f1 = 5
f2 = 70
npp = 2**15
fs = 30000

if scan:
    R = 1
    P = 1
    Avec = [10, 30, 50, 70, 80, 100, 120]
    Avec = [0.1, 1, 5, 10, 15, 30, 40, 50, 70, 80, 100, 120, 150]
    Avec = [0.1, 150]
    upsamp = 1
    fname = 'scan'
else:
    R = 2
    P = 6
    Avec = [20]
    upsamp = 20
    fname = 'ms'

ns = npp*R*P
t = np.arange(ns)/fs
fsint = upsamp * fs
nppint = upsamp * npp
# add extra period which will be removed due to edge effects
Pfilter = 1
if upsamp > 1:
    P = Pfilter + P
nsint = nppint*P*R
dt = 1/fsint
Ntr = 1

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

nlx = NLS(Tanhdryfriction(eps=eps, w=w))
nly = None

# cont time
a, b, c, d = mkc2ss(M, K, C)
csys = signal.StateSpace(a, b, c, d)
Ec = np.zeros((2*ndof, 1))
Fc = np.zeros((ndof, 0))
Ec[ndof+nldof] = -muN

# cmodel = NLSS(csys.A, csys.B, csys.C, csys.D, Ec, Fc)
# cmodel.add_nl(nlx=nlx, nly=nly)

# nhar = 1000
# np.random.seed(0)
# ufunc = multisine_time(f1, f2, N=nhar)
# def fex_cont(A, t):
#     t = np.atleast_1d(t)
#     fex = np.zeros((len(t), ndof))
#     fex[:, fdof] = A*ufunc(t)
#     return fex
#     #return np.vstack((np.zeros(len(t)), A*ufunc(t), np.zeros(len(t)))).T

# f0 = (f2-f1) / nhar
# t2 = P/f0
# tc = np.linspace(0, t2, nppint*P, endpoint=False)
# fsc = f0*nppint
# freqc = np.arange(nppint)/nppint * fsc

# x = ufunc(tc)
# X = np.fft.fft(x)
# nfd = X.shape[0]//2
# plt.figure()
# plt.plot(freq[:nfd], db(np.abs(X[:nfd])))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude (dB)')


# convert to discrete time
dsys = csys.to_discrete(dt=dt, method='zoh')  # tustin
Ed = np.zeros((2*ndof, 1))
Fd = np.zeros((ndof, 0))
# euler discretization
Ed[ndof+nldof] = -muN*dt

dmodel = NLSS(dsys.A, dsys.B, dsys.C, dsys.D, Ed, Fd)
dmodel.add_nl(nlx=nlx, nly=nly)

np.random.seed(0)
u, lines, freq = multisine(f1, f2, N=nppint, fs=fsint, R=R, P=P)
fext = np.zeros((nsint, ndof))

for A in Avec:
    print(f'Discrete started with ns: {nsint}, A: {A}')
    # Transient: Add periods before the start of each realization. To generate
    # steady state data.
    T1 = np.r_[npp*Ntr, np.r_[0:(R-1)*P*nppint+1:P*nppint]]
    fext[:, fdof] = A*u.ravel()
    _, y, x = dmodel.simulate(fext, T1=T1)
    # fexc = partial(fex_cont, A)
    # _, yc, xc = dnlsim2(cmodel, fexc, tc)

    try:
        if scan:
            # np.savez(f'data/scan_A{A}.npz', x=x, freq=freq, Ntint=Ntint,
            #          fdof=fdof, nldof=nldof)
            # plot frf for forcing and tanh node
            Yd = np.fft.fft(y[-nppint:, [fdof, nldof]], axis=0)
            #Yc = np.fft.fft(yc[-nppint:, [fdof, nldof]], axis=0)
            nfd = Yd.shape[0]//2
            plt.figure()
            plt.plot(freq[:nfd], db(np.abs(Yd[:nfd])))
            # plt.plot(freqc[:nfd], db(np.abs(Yc[:nfd])))
            plt.xlim([0, 50])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.legend(['d: Force dof', 'd: nl dof', 'c: Force dof', 'c: nl dof'])
            plt.title(f'A: {A}')
            plt.minorticks_on()
            plt.grid(which='both')
            # plt.savefig(f'fig/dc_b{benchmark}_A{A}_fft_comp_n{fdof}.png')
    except ValueError as e:
        print(f'Discrete stepping failed with error {e}. For A: {A}')

# We need to reshape into (npp,m,R,P)
fext = fext.reshape(R, P, nppint, ndof).transpose(2, 3, 0, 1)
# fext.shape = (Ntint,P,R,ndof)
y = y.reshape(R, P, nppint, ndof).transpose(2, 3, 0, 1)
x = x.reshape(R, P, nppint, ndof*2).transpose(2, 3, 0, 1)

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
