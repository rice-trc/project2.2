#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm

from pyvib.common import db
from pyvib.fnsi import FNSI
from pyvib.forcing import multisine
from pyvib.frf import covariance
from pyvib.nlss import NLSS
from pyvib.nonlinear_elements import Nonlinear_Element, Pnl, Tanhdryfriction
from pyvib.signal import Signal
from pyvib.subspace import Subspace

# data containers
Data = namedtuple('Data', ['sig', 'uest', 'yest', 'uval', 'yval', 'utest',
                           'ytest', 'um', 'ym', 'covY', 'freq', 'lines',
                           'npp', 'Ntr', 'fs'])
Result = namedtuple('Result', ['est_err', 'val_err', 'test_err', 'noise',
                               'nl_errvec', 'descrip'])
nmax = 100
info = 1
p = 2
weight = False
add_noise = False


def read_data(fname, Ntr=1, Rest=2):
    
    data = np.load(fname)
    u = data['fext2'][:,None]
    y = data['x2']
    lines = data['lines']
    fs = 750  #data['fs']
    npp, p, R, P = y.shape
    freq = np.arange(npp)/npp*fs

    ## START of Identification ##
    # partitioning the data. Use last period of two last realizations.
    # test for performance testing and val for model selection
    utest = u[:, :, -1, -1]
    ytest = y[:, :, -1, -1]
    uval = u[:, :, -2, -1]
    yval = y[:, :, -2, -1]
    # all other realizations are used for estimation
    uest = u[..., :Rest, :]
    yest = y[..., :Rest, :]
    # noise estimate over periods. This sets the performace limit for the
    # estimated model
    covY = covariance(yest)

    # create signal object
    sig = Signal(uest, yest, fs=fs)
    sig.lines = lines
    # plot periodicity for one realization to verify data is steady state
    # sig.periodicity()
    # Calculate BLA, total- and noise distortion. Used for subspace
    # identification
    sig.bla()
    # average signal over periods. Used for training of PNLSS model
    um, ym = sig.average()

    return Data(sig, uest, yest, uval, yval, utest, ytest, um, ym, covY,
                freq, lines, npp, Ntr, fs)

def identify(data, nlx, nly, nmax=25, info=2, fnsi=False):
    # transient: Add one period before the start of each realization. Note that
    # this is for the signal averaged over periods
    Rest = data.yest.shape[2]
    T1 = np.r_[data.npp*data.Ntr, np.r_[0:(Rest-1)*data.npp+1:data.npp]]

    linmodel = Subspace(data.sig)
    linmodel._cost_normalize = 1
    linmodel.estimate(2, 5, weight=weight)
    linmodel.optimize(weight=weight, info=info)

    # estimate NLSS
    model = NLSS(linmodel)
    # model._cost_normalize = 1
    model.add_nl(nlx=nlx, nly=nly)
    model.set_signal(data.sig)
    model.transient(T1)
    model.optimize(lamb=100, weight=weight, nmax=nmax, info=info)
    # get best model on validation data. Change Transient settings, as there is
    # only one realization
    nl_errvec = model.extract_model(data.yval, data.uval, T1=data.npp*data.Ntr,
                                    info=info)
    models = [linmodel, model]
    descrip = [type(mod).__name__ for mod in models]

    if fnsi:
        # FNSI can only use 1 realization
        sig = deepcopy(data.sig)
        # This is stupid, but unfortunately nessecary
        sig.y = sig.y[:, :, 0][:, :, None]
        sig.u = sig.u[:, :, 0][:, :, None]
        sig.R = 1
        sig.average()
        fnsi1 = FNSI()
        fnsi1.set_signal(sig)
        fnsi1.add_nl(nlx=nlx)
        fnsi1.estimate(n=2, r=5, weight=weight)
        fnsi1.transient(T1)
        fnsi2 = deepcopy(fnsi1)
        fnsi2.optimize(lamb=100, weight=weight, nmax=nmax, info=info)
        models = models + [fnsi1, fnsi2]
        descrip = descrip + ['FNSI', 'FNSI optimized']

    descrip = tuple(descrip)  # convert to tuple for legend concatenation
    # simulation error
    val = np.empty((*data.yval.shape, len(models)))
    est = np.empty((*data.ym.shape, len(models)))
    test = np.empty((*data.ytest.shape, len(models)))
    for i, model in enumerate(models):
        test[..., i] = model.simulate(data.utest, T1=data.npp*data.Ntr)[1]
        val[..., i] = model.simulate(data.uval, T1=data.npp*data.Ntr)[1]
        est[..., i] = model.simulate(data.um, T1=T1)[1]

    Pest = data.yest.shape[3]
    # convenience inline functions

    def stack(ydata, ymodel): return \
        np.concatenate(
        (ydata[..., None], (ydata[..., None] - ymodel)), axis=2)

    def rms(y): return np.sqrt(np.mean(y**2, axis=0))
    est_err = stack(data.ym, est)  # (npp*R,p,nmodels)
    val_err = stack(data.yval, val)
    test_err = stack(data.ytest, test)
    noise = np.abs(np.sqrt(Pest*data.covY.squeeze()))

    if info:
        print()
        print(f"err for models: signal, {descrip}")
        # print(f'rms error noise:\n{rms(noise)}     \ndb: \n{db(rms(noise))} ')
        # only print error for p = 0. Almost equal to p = 1
        print(f'rms error est (db): \n{db(rms(est_err[:,0]))}')
        print(f'rms error val (db): \n{db(rms(val_err[:,0]))}')
        # print(f'rms error test: \n{rms(test_err)}  \ndb: \n{db(rms(test_err))}')
    return Result(est_err, val_err, test_err, noise, nl_errvec, descrip)


#

fname = 'data/ms_condenced.npz'
Rest = 2
Ntr = 0
tahn1 = Tanhdryfriction(eps=0.1, w=[0,1])
nlx = [tahn1]


data = np.load(fname)
u = data['fext2'][:,None,:,1:]
y = data['x2']
lines = data['lines']
fs = 750  #data['fs']
npp, p, R, P = y.shape
freq = np.arange(npp)/npp*fs

## START of Identification ##
# partitioning the data. Use last period of two last realizations.
# test for performance testing and val for model selection
utest = u[:, :, -1, -1]
ytest = y[:, :, -1, -1]
uval = u[:, :, -2, -1]
yval = y[:, :, -2, -1]
# all other realizations are used for estimation
uest = u[..., :Rest, Ntr:]
yest = y[..., :Rest, Ntr:]
# noise estimate over periods. This sets the performace limit for the
# estimated model
covY = covariance(yest)

# create signal object
sig = Signal(uest, yest, fs=fs)
sig.lines = lines
# plot periodicity for one realization to verify data is steady state
# sig.periodicity(dof=16)
# Calculate BLA, total- and noise distortion. Used for subspace
# identification
sig.bla()
# average signal over periods. Used for training of PNLSS model
um, ym = sig.average()




dat1 = Data(sig, uest, yest, uval, yval, utest, ytest, um, ym, covY,
                freq, lines, npp, Ntr, fs)




plot_bla(dat1, 16)














def plot(res, data, p):
    figs = {}
    lines = data.lines
    freq = data.freq
    Pest = data.yest.shape[3]

    # result on validation data
    N = len(data.yval)
    freq = np.arange(N)/N*fs
    plottime = res.val_err
    plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
    plt.figure()
    plt.plot(freq[lines], db(plotfreq[lines, p]), '.')
    plt.plot(freq[lines], db(np.sqrt(Pest*data.covY[lines, p, p].squeeze() / N)),
             '.')
    plt.xlabel('Frequency')
    plt.ylabel('Output (errors) (dB)')
    plt.legend(('Output',) + res.descrip + ('Noise',))
    plt.title(f'Validation results p:{p}')
    figs['val_data'] = (plt.gcf(), plt.gca())

    # optimization path for NLSS
    plt.figure()
    plt.plot(db(res.nl_errvec))
    imin = np.argmin(res.nl_errvec)
    plt.scatter(imin, db(res.nl_errvec[imin]))
    plt.xlabel('Successful iteration number')
    plt.ylabel('Validation error [dB]')
    plt.title('Selection of the best model on a separate data set')
    figs['pnlss_path'] = (plt.gcf(), plt.gca())

    return figs

def plot_path(res, data, p):
    figs = {}
    # optimization path for NLSS
    plt.figure()
    plt.plot(db(res.nl_errvec))
    imin = np.argmin(res.nl_errvec)
    plt.scatter(imin, db(res.nl_errvec[imin]))
    plt.xlabel('Successful iteration number')
    plt.ylabel('Validation error [dB]')
    plt.title('Selection of the best model on a separate data set')
    figs['pnlss_path'] = (plt.gcf(), plt.gca())
    return figs

def plot_time(res, data, p):
    figs = {}
    plt.figure()
    plt.plot(res.est_err[:, p])
    plt.xlabel('Time index')
    plt.ylabel('Output (errors)')
    plt.legend(('Output',) + res.descrip)
    plt.title(f'Estimation results p:{p}')
    figs['estimation_error'] = (plt.gcf(), plt.gca())
    return figs

def plot_bla(data, p):
    figs = {}
    lines = data.lines
    freq = data.freq

    # BLA plot. We can estimate nonlinear distortion
    # total and noise distortion averaged over P periods and M realizations
    # total distortion level includes nonlinear and noise distortion
    plt.figure()
    # When comparing distortion(variance, proportional to power) with
    # G(propertional to amplitude(field)), there is two definations for dB:
    # dB for power: Lp = 10 log10(P).
    # dB for field quantity: Lf = 10 log10(F²)
    # Alternative calc: bla_noise = db(np.abs(sig.covGn[:,pp,pp])*R, 'power')
    # if the signal is noise-free, fix noise so we see it in plot
    bla_noise = db(np.sqrt(np.abs(data.sig.covGn[:, p, p])*R))
    bla_noise[bla_noise < -150] = -150
    bla_tot = db(np.sqrt(np.abs(data.sig.covG[:, p, p])*R))
    bla_tot[bla_tot < -150] = -150

    plt.plot(freq[lines], db(np.abs(data.sig.G[:, p, 0])))
    plt.plot(freq[lines], bla_noise, 's')
    plt.plot(freq[lines], bla_tot, '*')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('magnitude (dB)')
    plt.title(f'Estimated BLA and nonlinear distortion p: {p}')
    plt.legend(('BLA FRF', 'Noise Distortion', 'Total Distortion'))
    #plt.gca().set_ylim(bottom=-150)
    figs['bla'] = (plt.gcf(), plt.gca())
    return figs

def savefig(fname, figs):
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"{fname}{k}{i}.png")