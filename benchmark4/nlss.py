#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from scipy.io import loadmat

from pyvib.common import db
from pyvib.frf import covariance
from pyvib.nlss import NLSS
from pyvib.nonlinear_elements import (NLS, Pnlss, Polynomial, Polynomial_x,
                                      Tanhdryfriction, Unilatteralspring)
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""This script identifies a nonlinear state space model from data

For the identification, you need to attach nonlinear functions for which the
coefficients should be estimated. For general nonlinear data you can use
PNLSS. If you know the functional form of the nonlinearity, and it is
localized, then you can specify the nonlinearity in terms of the outputs.

The steps are
1) estimate a nonparametric linear model from input and output data
2) estimate a parametric linear state-space model from the nonparametric model
3) estimate the parameters in the full NLSS model
"""

# save figures to disk
savefig = False
add_noise = False
weight = False
fname = 'famp05'

nlfunc = 'pnlss'
#nlfunc = 'tahn'

data = loadmat(f'TRANSIENT/{fname}/CLCLEF_MULTISINE.mat')
f1 = data['f1'].item()
f2 = data['f2'].item()
df = data['df'].item()
fs = data['fsamp'].item()
fdof = data['fdof'].item()
lines = np.arange(f1//df+1,f2//df, dtype=int)

u = data['u'][...,None]  # shape (npp,P,R,m)
y = data['y']  # shape (npp,P,R,p)
yd = data['ydot']

# We need the shape to be (npp,p,R,P)
u = u.transpose(0,3,2,1)
y = y.transpose(0,3,2,1)
yd = yd.transpose(0,3,2,1)

# use the response at the excitation node (-1 for python being 0-index)
pdof = fdof - 1
y = np.stack((y[:,pdof], yd[:,pdof]),axis=1)
# if you want to use multiple outputs for the ID, use this instead
# pdofs = np.array([1,2,...])
# y = np.concatenate((y[:,pdofs], yd[:,pdofs]),axis=1)

# Add colored noise to the output. randn generate white noise
if add_noise:
    np.random.seed(10)
    noise = 1e-3*np.std(y[:,-1,-1]) * np.random.randn(*y.shape)
    # Do some filtering to get colored noise
    noise[1:-2] += noise[2:-1]
    y += noise

## START of Identification ##
# partitioning the data. Use last period of two last realizations.
# test for performance testing and val for model selection
utest = u[:,:,-1,-1]
ytest = y[:,:,-1,-1]
uval = u[:,:,-2,-1]
yval = y[:,:,-2,-1]
# all other realizations are used for estimation
uest = u[...,:-2,:]
yest = y[...,:-2,:]
# noise estimate over periods. This sets the performace limit for the estimated
# model
covY = covariance(yest)
npp, p, Rest, Pest = yest.shape
npp, m, Rest, Pest = uest.shape
Ptr = 5  # number of periods to use for transient handling during simulation

# create signal object
sig = Signal(uest,yest,fs=fs)
sig.lines = lines
# plot periodicity for one realization to verify data is steady state
# sig.periodicity()
# Calculate BLA, total- and noise distortion. Used for subspace identification
sig.bla()
# average signal over periods. Used for training of PNLSS model
um, ym = sig.average()

# model orders and Subspace dimensioning parameter
nvec = [3]
maxr = 10


if 'linmodel' not in locals() or True:
    linmodel = Subspace(sig)
    linmodel.estimate(2, maxr, weight=weight)
    linmodel.optimize(weight=weight)
    
    print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")
    linmodel_orig = linmodel

if False:  # dont scan subspace
    linmodel = Subspace(sig)
    # get best model on validation data
    models, infodict = linmodel.scan(nvec, maxr, weight=weight)
    l_errvec = linmodel.extract_model(yval, uval)
    # or estimate the subspace model directly
    linmodel.estimate(2, 5, weight=weight)  # best model, when noise weighting is used
    linmodel.optimize(weight=weight)
    print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")
    
linmodel = deepcopy(linmodel_orig)

# estimate PNLSS
# transient: Add one period before the start of each realization. Note that
# this is for the signal averaged over periods
T1 = np.r_[npp*Ptr, np.r_[0:(Rest-1)*npp+1:npp]]

# select nonlinear functions
nly = None
pnlss1 = Pnlss(degree=[2,3], structure='full')
w = [0,1]
tahn1 = Tanhdryfriction(eps=0.01, w=w)
if nlfunc.casefold() == 'pnlss':
    nlx = NLS([pnlss1])
    nly = NLS([pnlss1])
elif nlfunc.casefold() == 'tahn':
    nlx = NLS([pnlss1, tahn1])
    nlx = NLS([tahn1])




model = NLSS(linmodel)
model.add_nl(nlx=nlx, nly=nly)
model.set_signal(sig)
model.transient(T1)
model.optimize(lamb=100, weight=weight, nmax=25)

#raise SystemExit(0)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=npp)

models = [linmodel, model]
descrip = [type(mod).__name__ for mod in models]
descrip = tuple(descrip)  # convert to tuple for legend concatenation in figs
# simulation error
val = np.empty((*yval.shape, len(models)))
est = np.empty((*ym.shape, len(models)))
test = np.empty((*ytest.shape, len(models)))
for i, model in enumerate(models):
    test[...,i] = model.simulate(utest, T1=npp*Ptr)[1]
    val[...,i] = model.simulate(uval, T1=npp*Ptr)[1]
    est[...,i] = model.simulate(um, T1=T1)[1]

# convenience inline functions
stack = lambda ydata, ymodel: \
    np.concatenate((ydata[...,None], (ydata[...,None] - ymodel)),axis=2)
rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = stack(ym, est)  # (npp*R,p,nmodels)
val_err = stack(yval, val)
test_err = stack(ytest, test)
noise = np.abs(np.sqrt(Pest*covY.squeeze()))
print(f"### err for models {descrip} ###")
print(f'rms error noise: \n{rms(noise)}     \ndb: \n{db(rms(noise))} ')
print(f'rms error est:   \n{rms(est_err)}   \ndb: \n{db(rms(est_err))}')
print(f'rms error val:   \n{rms(val_err)}   \ndb: \n{db(rms(val_err))}')
print(f'rms error test:  \n{rms(test_err)}  \ndb: \n{db(rms(test_err))}')


## Plots ##
# store figure handle for saving the figures later
figs = {}

# linear and nonlinear model error; plot for each output
for pp in range(p):
    plt.figure()
    plt.plot(est_err[:,pp])
    plt.xlabel('Time index')
    plt.ylabel('Output (errors)')
    plt.legend(('Output',) + descrip)
    plt.title(f'Estimation results p:{pp}')
    figs['estimation_error'] = (plt.gcf(), plt.gca())

    # result on validation data
    N = len(yval)
    freq = np.arange(N)/N*fs
    plottime = val_err
    plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
    nfd = plotfreq.shape[0]
    plt.figure()
    plt.plot(freq[lines], db(plotfreq[lines,pp]), '.')
    plt.plot(freq[lines], db(np.sqrt(Pest*covY[lines,pp,pp].squeeze() / N)), '.')
    plt.xlabel('Frequency')
    plt.ylabel('Output (errors) (dB)')
    plt.legend(('Output',) + descrip + ('Noise',))
    plt.title(f'Validation results p:{pp}')
    figs['val_data'] = (plt.gcf(), plt.gca())

    # result on test data
    N = len(ytest)
    freq = np.arange(N)/N*fs
    plottime = test_err
    plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
    nfd = plotfreq.shape[0]
    plt.figure()
    plt.plot(freq[:nfd//2], db(plotfreq[:nfd//2,pp]), '.')
    plt.plot(freq[:nfd//2], db(np.sqrt(Pest*covY[:nfd//2,pp,pp].squeeze() / N)), '.')
    plt.xlabel('Frequency')
    plt.ylabel('Output (errors) (dB)')
    plt.legend(('Output',) + descrip + ('Noise',))
    plt.title(f'Test results p:{pp}')
    figs['test_data'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
plt.plot(db(nl_errvec))
imin = np.argmin(nl_errvec)
plt.scatter(imin, db(nl_errvec[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')
figs['pnlss_path'] = (plt.gcf(), plt.gca())

# subspace plots
#figs['subspace_optim'] = linmodel.plot_info()
#figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/tutorial_{k}{i}.pdf")

plt.show()


"""
Workable parameters
-------------------
RMSu = 0.05
Ntr = 5
E = np.array([[3.165156145e-03],
             [2.156132115e-03]])
nlx = NLS([Tanhdryfriction(eps=0.1, w=[1])])

----
RMSu = 0.05
Ntr = 5
E = np.array([[3.165156145e-03],
             [2.156132115e-03]])
gap = 0.25
nlx = NLS([Unilatteralspring(gap=gap, w=[1])])
----

RMSu = 0.05
Ntr = 5
E = Efull[:,:2]
nlx = NLS([poly2y, poly1y])  #, poly3])

nlx2 = NLS([poly1y,poly3y,poly2x,poly2y])  #,poly3])
nly2 = None

====
p = 2

if p == 2:
    C = np.vstack((C,C))
    D = np.vstack((D,0.1563532))

E = Efull
F = Ffull
nlx = NLS([Pnlss(degree=[2,3], structure='full')])
nly = NLS([Pnlss(degree=[2,3], structure='full')])
----

elif p ==2:
    Wy = np.array([[1,0],[0,1]])
    exp1 = [2,1]
    exp2 = [2,2]
    exp3 = [3,1]


nly = None
nlx = NLS([poly2y, poly1y])
E = Efull[:,:len(nlx.nls)]

nlx2 = NLS([poly1y,poly3y,poly2x,poly2y])
"""
