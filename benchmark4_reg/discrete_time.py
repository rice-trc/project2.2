#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat

from pyvib.common import db, dsample
from pyvib.forcing import multisine, multisine_time
from pyvib.modal import mkc2ss
from pyvib.newmark import Newmark
from pyvib.nlss import NLSS, nlsim2
from pyvib.nonlinear_elements import NLS, Polynomial, Tanhdryfriction
from pyvib.nonlinear_elements_newmark import NLS as nmNLS
from pyvib.nonlinear_elements_newmark import Polynomial as nmPolynomial
from pyvib.nonlinear_elements_newmark import \
    Tanhdryfriction as nmTanhdryfriction

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

We need at least 2**13(8192) points per period for good identification of the
linear system. Even if the system is sampled with more points
"""

scan = False
benchmark = 4

# define multisine
f1 = 5
f2 = 100
npp = 2**14
fs = 700
if scan:
    R = 1
    P = 2
    Avec = [10, 30, 50, 70, 80, 100, 120]
    Avec = [0.1, 1, 5, 10, 15, 30, 40, 50, 70, 80, 100, 120, 150]
    #Avec = [0.1, 50, 1500, 3000]
    Avec = np.round(np.logspace(1, 4, 20)).astype(int)
    upsamp = 70
    fname = 'scan'
else:
    R = 2
    P = 6
    Avec = [700]
    Avec = [700]
    upsamp = 70  #70
    fname = 'ms'
    fname = 'pol'

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

eps = 0.1
wd = [0,0,0,0,0,1]
nlx = NLS(Tanhdryfriction(eps=eps, w=wd))

muN = 1e9
exponent = 3
wd = [0,0,1,0,0,0]
wd = w
nlx = NLS(Polynomial(exponent=exponent, w=wd))
# nlx = None
nly = None
epsf = f'{eps}'.replace('.', '')

# cont time
a, b, c, d = mkc2ss(M, K, C)
fact = 1
# include velocity in output
if len(wd) == 6:
    c = np.vstack((c ,np.hstack((np.zeros((3,3)), np.eye(3))) ))
    d = np.vstack((d, np.zeros((3,3))))
    fact = 2
csys = signal.StateSpace(a, b, c, d)
Ec = np.zeros((2*ndof, 1))
Fc = np.zeros((fact*ndof, 0))
Ec[ndof+nldof] = -muN

cmodel = NLSS(csys.A, csys.B, csys.C, csys.D, Ec, Fc)
cmodel.add_nl(nlx=nlx, nly=nly)


def fex_cont(A, u, t):
    t = np.atleast_1d(t)
    fex = np.zeros((len(t), ndof))
    fex[:, fdof] = A*u(t)
    return fex


def simulate_cont(sys, A, t):
    nt = len(t)
    y = np.empty((R, nt, sys.outputs))
    x = np.empty((R, nt, len(sys.A)))
    u = np.empty((R, nt))
    for r in range(R):
        np.random.seed(r)
        ufunc, lines = multisine_time(f1, f2, N=nhar)
        fexc = partial(fex_cont, A, ufunc)

        _, yr, xr = nlsim2(sys, fexc, t=tc)
        y[r] = yr
        x[r] = xr
        u[r] = ufunc(t)

    return y.reshape((R*nt, -1)), x.reshape((R*nt, -1)), u, lines


nhar = 1000
f0 = (f2-f1) / nhar
t2 = P/f0
tc = np.linspace(0, t2, nppint*P, endpoint=False)
fsc = f0*nppint
freqc = np.arange(nppint)/nppint * fsc

# convert to discrete time
dsys = csys.to_discrete(dt=dt, method='foh')  # tustin
Ed = np.zeros((2*ndof, 1))
Fd = np.zeros((fact*ndof, 0))
# euler discretization
Ed[ndof+nldof] = -muN*dt

dmodel = NLSS(dsys.A, dsys.B, dsys.C, dsys.D, Ed, Fd, dt=dsys.dt)
dmodel.add_nl(nlx=nlx, nly=nly)

# newmark
nls = nmNLS(nmTanhdryfriction(eps=eps, w=w, kt=muN))
nls = nmNLS(nmPolynomial(exp=exponent, w=w, k=muN))

# nls = None
sys = Newmark(M, C, K, nls)
nm = False

np.random.seed(0)
ud, linesd, freqd = multisine(f1, f2, N=nppint, fs=fsint, R=R, P=P)
fext = np.zeros((nsint, ndof))

for A in Avec:
    print(f'Discrete started with ns: {nsint}, A: {A}, R: {R}, P: {P}, '
          f'upsamp: {upsamp}, eps:{eps}')
    # Transient: Add periods before the start of each realization. To generate
    # steady state data.
    T1 = np.r_[npp*Ntr, np.r_[0:(R-1)*P*nppint+1:P*nppint]]
    fext[:, fdof] = A*ud.ravel()
    _, yd, xd = dmodel.simulate(fext, T1=T1)
    yc, xc, uc, linesc = simulate_cont(cmodel, A, tc)

    try:
        ynm, ydnm, yddnm = sys.integrate(fext, dt, x0=None, v0=None,
                                         sensitivity=False)
        Ynm = np.fft.fft(ynm[-nppint:, [fdof, nldof]], axis=0)
        nm = True
    except ValueError as e:
        print(f'Discrete stepping failed with error {e}. For A: {A}')

    #if scan:
    # plot frf for forcing and tanh node
    Yd = np.fft.fft(yd[-nppint:, [fdof, nldof]], axis=0)
    Yc = np.fft.fft(yc[-nppint:, [fdof, nldof]], axis=0)
    nfd = Yd.shape[0]//2
    plt.figure()
    plt.plot(freqd[:nfd], db(np.abs(Yd[:nfd])))
    if nm:
        plt.plot(freqd[:nfd], db(np.abs(Ynm[:nfd])))
        nm = False
    plt.plot(freqc[:nfd], db(np.abs(Yc[:nfd])))
    plt.xlim([0, 50])
    plt.ylim(bottom=-150)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend(['d: Force dof', 'd: nl dof', 'nm: Force dof', 'nm: nl dof',
                'c: Force dof', 'c: nl dof'])
    plt.title(f'A: {A}')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.savefig(f'fig/dc_b{benchmark}_A{A}_eps{epsf}_fft_comp_n{fdof}.png')

# We need to reshape into (npp,m,R,P)
if len(wd) != 6:
    yd = np.hstack((yd,yd))
    yc = np.hstack((yc,yc))
ys = [ynm, ydnm, yddnm, yd[:,:3], yd[:,3:], yc[:,:3], yc[:,3:]]
ys = [y.reshape(R, P, nppint, ndof).transpose(2, 3, 0, 1) for y in ys]

xs = [xd, xc]
xs = [x.reshape(R, P, nppint, 2*ndof).transpose(2, 3, 0, 1) for x in xs]

us = [A*ud, uc]
us = [u.reshape(R, P, nppint, 1).transpose(2, 3, 0, 1) for u in us]

if upsamp:  # > 1:
    ys = [dsample(y, upsamp, zero_phase=True) for y in ys]
    xs = [dsample(y, upsamp, zero_phase=True) for y in xs]
    us = [u[::upsamp, :, :, :] for u in us]

fname = f'data/{fname}_A{A}_upsamp{upsamp}_fs{fs}_eps{epsf}.npz'
np.savez(fname,
         ynm=ys[0], ydotnm=ys[1], yddotnm=ys[2],
         yd=ys[3], ydotd=ys[4], xd=xs[0], ud=us[0], linesd=linesd,
         yc=ys[5], ydotc=ys[6], xc=xs[1], uc=us[1], linesc=linesc,
         fs=fs, A=A, fsc=f0*npp)
print(f'data saved as {fname}')

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
# x = ufunc(tc)
# X = np.fft.fft(x)
# nfd = X.shape[0]//2
# plt.figure()
# plt.plot(freq[:nfd], db(np.abs(X[:nfd])))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude (dB)')


# plt.show()
