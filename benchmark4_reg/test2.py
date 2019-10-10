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

"""Identify coefficient of tanh nonlinearity
fnl = k*tanh(ẏ/eps)
where eps is the regularization parameter specified a priory

Depending on the value of eps, we get a good or bad identification. If we look
at the BLA plot, we see that the noise level is high for the values of eps that
result in poor estimation. For 'good' eps the noise floor is zero.

So somehow certain values of eps result in data that seems to be disturbed by
noise, which I think is a sign of unsteady data.
The data is generated as noise-free

fnsi good:
eps = 0.1, 0.0001

fnsi bad:
eps = 0.01
"""

# data containers
Data = namedtuple('Data', ['sig', 'uest', 'yest', 'uval', 'yval', 'utest',
                           'ytest', 'um', 'ym', 'covY', 'freq', 'lines',
                           'npp', 'Ntr'])
Result = namedtuple('Result', ['est_err', 'val_err', 'test_err', 'noise',
                               'errvec', 'descrip'])
p = 2
weight = False
add_noise = False
## Generate data from true model ##
# Construct model to estimate
A = np.array([[0.73915535, -0.62433133], [0.6247377, 0.7364469]])
B = np.array([[0.79287245], [-0.34515159]])
C = np.array([[0.71165154, 0.34917771]])
D = np.array([[0.04498052]])
if p == 2:
    C = np.vstack((C, C))
    D = np.vstack((D, 0.1563532))

Ffull = np.array([
    [-0.00867042, -0.00636662,  0.00197873, -0.00090865, -0.00088879,
     -0.02759694, -0.01817546, -0.10299409,  0.00648549,  0.08990175,
     0.21129849,  0.00030216,  0.03299013,  0.02058325, -0.09202439,
     -0.0380775],
    [-0.17323214, -0.08738017, -0.11346953, -0.08077963, -0.05496476,
     0.01874564, -0.02946581, -0.01869213, -0.07492472,  0.06868484,
     -0.02770704,  0.19900055, -0.089364, -0.00410125,  0.13002691,
     -0.11460958]])
Efull = np.array([
    [1.88130305e-01, -2.70291900e-01,  9.12423046e-03, -5.78088500e-01,
     9.54588221e-03,  5.08576019e-04, -1.33890850e+00, -2.02171960e+00,
     -4.05918956e-01, -1.37744223e+00,  1.21206232e-01, -9.26349423e-02,
     -5.38072197e-01,  2.34134460e-03,  4.94334690e-02, -1.88329572e-02],
    [-5.35196110e-01, -3.66250013e-01,  2.34622651e-02,  1.43228677e-01,
     -1.35959331e-02,  1.32052696e-02,  7.98717915e-01,  1.35344901e+00,
     -5.29440815e-02,  4.88513652e-01,  7.81285093e-01, -3.41019453e-01,
     2.27692972e-01,  7.70150211e-02, -1.25046731e-02, -1.62456154e-02]])

# excitation signal
RMSu = 0.05     # Root mean square value for the input signal
R = 4           # Number of phase realizations (one for validation and one
# for testing)
P = 3           # Number of periods
kind = 'Full'  # 'Full','Odd','SpecialOdd', or 'RandomOdd': kind of multisine
m = D.shape[1]  # number of inputs
p = C.shape[0]  # number of outputs
fs = 1          # normalized sampling rate


def simulate(true_model, npp=1024, Ntr=1, add_noise=False):
    print()
    print(f'Nonlinear parameters:',
          f'{len(true_model.nlx.active) + len(true_model.nly.active)}')
    print(f'Parameters to estimate: {true_model.npar}')
    # set non-active coefficients to zero. Note order of input matters
    idx = np.setdiff1d(np.arange(true_model.E.size), true_model.nlx.active)
    idy = np.setdiff1d(np.arange(true_model.F.size), true_model.nly.active)
    true_model.E.flat[idx] = 0
    true_model.F.flat[idy] = 0

    # get predictable random numbers. https://dilbert.com/strip/2001-10-25
    np.random.seed(10)
    # shape of u from multisine: (R,P*npp)
    u, lines, freq = multisine(N=npp, P=P, R=R, lines=kind, rms=RMSu)

    # Transient: Add Ntr periods before the start of each realization. To
    # generate steady state data.
    T1 = np.r_[npp*Ntr, np.r_[0:(R-1)*P*npp+1:P*npp]]
    _, yorig, _ = true_model.simulate(u.ravel(), T1=T1)
    u = u.reshape((R, P, npp)).transpose((2, 0, 1))[:, None]  # (npp,m,R,P)
    y = yorig.reshape((R, P, npp, p)).transpose((2, 3, 0, 1))

    # Add colored noise to the output. randn generate white noise
    if add_noise:
        np.random.seed(10)
        noise = 1e-3*np.std(y[:, -1, -1]) * np.random.randn(*y.shape)
        # Do some filtering to get colored noise
        noise[1:-2] += noise[2:-1]
        y += noise

    return {'y': y, 'u': u, 'lines': lines, 'freq': freq}


def partion_data(data, Rest=2, Ntr=1):
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
                freq, lines, npp, Ntr)


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


def identify_fnsi(data, nlx, nly, n, r, nmax=25, optimize=True, info=2):
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


def identify_linear(data, n, r, subscan=True, info=2):
    lin_errvec = []
    linmodel = Subspace(data.sig)
    linmodel._cost_normalize = 1
    if subscan:
        linmodel.scan(nvec=[2, 3, 4, 5, 6, 7, 8], maxr=20,
                      optimize=True, weight=False, info=info)
        lin_errvec = linmodel.extract_model(data.yval, data.uval)
        print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")

        #linmodel.estimate(n=n, r=r, weight=weight)
        #linmodel.optimize(weight=weight, info=info)
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
    figs['val_data'] = (plt.gcf(), plt.gca())

    return figs


def plot_path(res, data, p):
    figs = {}
    # optimization path for NLSS
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


def plot_bla(res, data, p):
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
    plt.gca().set_ylim(bottom=-150)
    figs['bla'] = (plt.gcf(), plt.gca())
    return figs


def savefig(fname, figs):
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"{fname}{k}{i}.png")


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

    nly_pnl = [Pnl(degree=[2, 3, 5], structure='statesonly')]
    nlx_pnl = [Pnl(degree=[2, 3, 5], structure='statesonly')]
    # models['nlss_pnl'], errvec['nlss_pnl'] = identify_nlss(
    #     data, models['lin'], nlx_pnl, nly_pnl, nmax=nmax, info=info)
    res = evaluate_models(data, models, errvec, info=info)
    return models, res


def disp_plot(data, res, nldof):
    f1 = plot_bla(res, data, nldof)
    f2 = plot_val(res, data, nldof)
    f3 = plot_path(res, data, nldof)
    f4 = plot_time(res, data, nldof)
    figs = {**f1, **f2, **f3, **f4}
    return figs

# parameters
nmax = 100
info = 1
subscan = False
nldof = 1

tahn1 = Tanhdryfriction(eps=0.0001, w=[0, 1])
nlx = [tahn1]
F = np.array([])
nly = None

# We get good identification using BLA
E = 1e-1*Efull[:, :len(nlx)]

true_model = NLSS(A, B, C, D, E, F)
true_model.add_nl(nlx=nlx, nly=nly)
raw_data3 = simulate(true_model, npp=2048, Ntr=2)
#raw_data3 = np.load('data/test.npz')

# Ntr: how many transient periods in T1 for identification
data3 = partion_data(raw_data3, Ntr=1)
# plot bla to see nonlinear distortion. Check noise level!
plot_bla([], data3, nldof)

models, res3 = identify(data3, nlx, nly, n=2, r=5, subscan=subscan)
figs = disp_plot(data3, res3, nldof)

# subspace plots
linmodel = models['lin']
figs['subspace_models'] = linmodel.plot_models()
if subscan:
    figs['subspace_optim'] = linmodel.plot_info()

# plot periodicity for one realization to verify data is steady state
figs['per'] = data3.sig.periodicity(dof=nldof)

savefig('fig/test_', figs)
