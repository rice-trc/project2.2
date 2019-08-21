#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
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
                           'npp', 'Ntr'])
Result = namedtuple('Result', ['est_err', 'val_err', 'test_err', 'noise',
                               'errvec', 'descrip'])
nmax = 100
info = 1
p = 2
weight = False
add_noise = False


def plot_bla(res, data, p):
    figs = {}
    lines = data.lines
    freq = data.freq
    npp, _, R, P = data.yest.shape

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
    # plt.gca().set_ylim(bottom=-150)
    figs['bla'] = (plt.gcf(), plt.gca())
    return figs


def plot_val(res, data, p):
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
    figs['val_error'] = (plt.gcf(), plt.gca())

    return figs


def plot_path(res, data, p):
    figs = {}
    plt.figure()
    for desc, err in res.errvec.items():
        if len(err) == 0:
            continue
        # optimization path for NLSS
        plt.plot(db(err), label=desc)
        imin = np.argmin(err)
        plt.scatter(imin, db(err[imin]))
    plt.xlabel('Successful iteration number')
    plt.ylabel('Validation error [dB]')
    plt.title('Selection of the best model on a separate data set')
    plt.legend()
    figs['path'] = (plt.gcf(), plt.gca())
    return figs


def plot_time(res, data, p):
    figs = {}
    plt.figure()
    plt.plot(res.est_err[:, p])
    plt.xlabel('Time index')
    plt.ylabel('Output (errors)')
    plt.legend(('Output',) + res.descrip)
    plt.title(f'Estimation results p:{p}')
    figs['est_error'] = (plt.gcf(), plt.gca())
    return figs


def disp_plot(data, res, nldof):
    f1 = plot_bla(res, data, nldof)
    f2 = plot_val(res, data, nldof)
    f3 = plot_path(res, data, nldof)
    f4 = plot_time(res, data, nldof)
    figs = {**f1, **f2, **f3, **f4}
    return figs


def savefig(fname, figs):
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"{fname}{k}{i}.png")


def identify_nlss(data, linmodel, nlx, nly, nmax=25, info=2):
    Rest = data.yest.shape[2]
    T1 = np.r_[data.npp*data.Ntr, np.r_[0:(Rest-1)*data.npp+1:data.npp]]
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

    return model, nl_errvec


def identify_fnsi(data, nlx, nly, n=6, r=15, nmax=25, optimize=True, info=2):
    fnsi_errvec = []
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
    fnsi1.estimate(n=n, r=r, weight=weight)
    fnsi1.transient(T1=data.npp*data.Ntr)
    if optimize:
        try:
            fnsi1.optimize(lamb=100, weight=weight, nmax=nmax, info=info)
            fnsi_errvec = fnsi1.extract_model(data.yval, data.uval,
                                              T1=data.npp*data.Ntr, info=info)
        except ValueError as e:
            print(f'FNSI optimization failed with {e}')
    return fnsi1, fnsi_errvec


def identify_linear(data, n=6, r=20, subscan=True, info=2):
    lin_errvec = []
    linmodel = Subspace(data.sig)
    #linmodel._cost_normalize = 1
    if subscan:
        linmodel.scan(nvec=[2, 3, 4, 5, 6, 7, 8], maxr=20,
                      optimize=True, weight=False, info=info)
        lin_errvec = linmodel.extract_model(data.yval, data.uval)
        print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")

        linmodel.estimate(n=n, r=r, weight=weight)
        linmodel.optimize(weight=weight, info=info)
    else:
        linmodel.estimate(n=n, r=r, weight=weight)
        linmodel.optimize(weight=weight, info=info)
    return linmodel, lin_errvec


def evaluate_models(data, models, errvec, info=2):

    descrip = tuple(models.keys())  # convert to tuple for legend concatenation
    models = list(models.values())
    Rest = data.yest.shape[2]
    T1 = np.r_[data.npp*data.Ntr, np.r_[0:(Rest-1)*data.npp+1:data.npp]]
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
    return Result(est_err, val_err, test_err, noise, errvec, descrip)


def partion_data(data, Rest=2, Ntr=1, Ntr_steady=1):
    y = data['y']
    u = data['u']
    lines = data['lines']
    freq = data['freq']
    npp, p, R, P = y.shape
    # partitioning the data. Use last period of two last realizations.
    # test for performance testing and val for model selection
    utest = u[:, :, -1, -1]
    ytest = y[:, :, -1, -1]
    uval = u[:, :, -2, -1]
    yval = y[:, :, -2, -1]
    # all other realizations are used for estimation
    uest = u[..., :Rest, Ntr_steady:]
    yest = y[..., :Rest, Ntr_steady:]
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
                freq, lines, npp, Ntr)


def identify(data, nlx, nly, n, r, subscan=True):
    errvec = {}
    models = {}
    models['lin'], _ = identify_linear(
        data, n=n, r=r, subscan=subscan, info=info)
    models['fnsi'], _ = identify_fnsi(
        data, nlx, nly, n=n, r=r, nmax=nmax, optimize=False, info=info)
    models['fnsi optim'], errvec['fnsi'] = identify_fnsi(
        data, nlx, nly, n=n, r=r, nmax=nmax, optimize=True, info=info)
    models['nlss'], errvec['nlss'] = identify_nlss(
        data, models['lin'], nlx, nly, nmax=nmax, info=info)

    # nly_pnl = [Pnl(degree=[2, 3, 5], structure='statesonly')]
    # nlx_pnl = [Pnl(degree=[2, 3, 5], structure='statesonly')]
    # models['nlss_pnl'], errvec['nlss_pnl'] = identify_nlss(
    # data, models['lin'], nlx_pnl, nly_pnl, nmax=nmax, info=info)
    res = evaluate_models(data, models, errvec, info=info)
    return models, res


def load_npz(fname, dtype='nm', include_vel=True):
    with np.load(fname) as data:
        if dtype == 'nm':
            d = {'lines': data['linesd'], 'fs': data['fs'].item(), 'u':
                 data['ud'], 'y': data['ynm'], 'yd': data['ydotnm'], 'ydd':
                 data['yddotnm']}
        elif dtype == 'discrete':
            d = {'lines': data['linesd'], 'fs': data['fs'].item(), 'u':
                 data['ud'], 'y': data['yd'], 'yd': data['ydotd']}
            # , 'yd': data['xd'][:, 3:]}

    npp = d['y'].shape[0]
    d['freq'] = np.arange(npp)/npp * d['fs']
    if include_vel:
        d['y'] = np.hstack((d['y'], d['yd'][:, -1][:, None]))

    return d


def load_mat(fname):
    d = {}
    lines = data['lines'].squeeze()[2:-1] - 1
    # from matlab: (npp,P,R,p). We need (npp,p,R,P)
    y = data['y'].squeeze().transpose(0, 3, 2, 1)
    ydot = data['ydot'].squeeze().transpose(0, 3, 2, 1)
    u = data['u'].squeeze()[..., None].transpose(0, 3, 2, 1)
    fs = data['fs'].item()

    return d


nldof = 2
subscan = False
dtype = 'discrete'

w = [0, 0, 0, 1]
eps = 0.1
tahn1 = Tanhdryfriction(eps=eps, w=w)
nlx = [tahn1]
nly = None

include_vel = True
Avec = [700]
fs = 750
fname = 'ms'
upsamp = 70

for A in Avec:
    print(f'A:{A}')
    # try:
    datname = f'data/{fname}_A{A}_upsamp{upsamp}_fs{fs}.npz'
    raw_data = load_npz(datname, dtype=dtype, include_vel=True)
    # Ntr: how many transient periods in T1 for identification
    data = partion_data(raw_data, Ntr=2, Ntr_steady=1)
    plot_bla([], data, nldof)

    models, res = identify(data, nlx, nly, n=6, r=15, subscan=subscan)
    figs = disp_plot(data, res, nldof)

    # subspace plots
    linmodel = models['lin']
    figs['subspace_models'] = linmodel.plot_models()
    if subscan:
        figs['subspace_optim'] = linmodel.plot_info()

    # plot periodicity for one realization to verify data is steady state
    figs['per'] = data.sig.periodicity(dof=nldof)

    savefig('fig/nlss_', figs)

    # except ValueError as e:
    #    print(f'Could not load {datname}. Error {e}')
