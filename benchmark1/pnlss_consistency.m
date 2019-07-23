clc
clear all
addpath('../src/matlab/')

%%
% famp = 150;
% linmodel   = load(sprintf('./data/ode45_multisine_f%d.mat',famp), 'model', 'fs', 'A');
% pnlssmodel = load(sprintf('./data/pnlssout_f%d.mat',famp), 'model');

Alevel = 25;
fs = 16384;
fs = 4096
linmodel   = load(sprintf('./data/ode45_multisine_A%d_F%d.mat',Alevel,fs), 'model');
pnlssmodel = load(sprintf('./data/pnlssout_A%d_F%d.mat',Alevel,fs), 'model');

lm = ss(linmodel.model.A, linmodel.model.B, linmodel.model.C, linmodel.model.D)

% exact discrete-continuous conversion
pm = d2c(ss(pnlssmodel.model.A, pnlssmodel.model.B, pnlssmodel.model.C, pnlssmodel.model.D, 1/fs));
[Ap, Bp, Cp, Tp] = ss2phys(pm.A, pm.B, pm.C);

% % numerical discrete-continuous conversion
% apm = ss((pnlssmodel.model.A-eye(2))*fs, pnlssmodel.model.B*fs,...
%     pnlssmodel.model.C, pnlssmodel.model.D);
% [Ap, Bp, Cp, Tp] = ss2phys(apm.A, apm.B, apm.C);

pmphy = ss(Ap, Bp, Cp, pm.D)
Bp(1)
Bp(2)

%% Converting continuous time model to PNLSS output SS model
Xpowers = pnlssmodel.model.xpowers(:,1:2);

% Nonlinear coefficients
linmodel.E = zeros(2, size(Xpowers,1));
linmodel.E(2, Xpowers(:,1)==3 & Xpowers(:,2)==0) = linmodel.model.nlcof.coef;
linmodel.Tmat = NLCOEF_TFMMATS(Tp, Xpowers);

PsiMat = NLCOEF_TFMMATS(Tp, Xpowers);

Elp = -inv(Tp)*linmodel.E*PsiMat  % In continuous time
Elpd = -inv(Tp)*linmodel.E*PsiMat/fs   % In discrete time

Elpd./pnlssmodel.model.E

%% Converting discrete PNLSS output to physical SS model
Xpowers = pnlssmodel.model.xpowers(:,1:2);

PsiMat = NLCOEF_TFMMATS(inv(Tp), Xpowers);

Epn = (Tp*pnlssmodel.model.E*PsiMat)*fs

Elin = log10(linmodel.E)
Epnl = log10(abs(Epn))
Xpowers'

%% Converting continuous time model to continuous time modal doman and then discretized

% Continuous linear part
[Philin, Dlin] =  eig(lm.A);
PsiMatlin = NLCOEF_TFMMATS(Philin, Xpowers);

Elmodallin_c = -inv(Philin)*linmodel.E*PsiMatlin;
Elmodallin_d = Elmodallin_c/fs;


% Discrete linear part (PNLSS)
[Phipn, Dpn] = eig(pnlssmodel.model.A);
PsiMatpn = NLCOEF_TFMMATS(Phipn, Xpowers);

Elmodalpn_d = inv(Phipn)*pnlssmodel.model.E*PsiMatpn
