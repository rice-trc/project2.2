clc
clear all

%%
famp = 150;
linmodel   = load(sprintf('./data/ode45_multisine_f%d.mat',famp), 'model', 'fs', 'A');
pnlssmodel = load(sprintf('./data/pnlssout_f%d.mat',famp), 'model');

lm = ss(linmodel.model.A, linmodel.model.B, linmodel.model.C, linmodel.model.D)
pm = d2c(ss(pnlssmodel.model.A, pnlssmodel.model.B, pnlssmodel.model.C, pnlssmodel.model.D, 1/linmodel.fs));

[Ap, Bp, Cp, Tp] = ss2phys(pm.A, pm.B, pm.C);
% Bp = Bp/linmodel.A;
% Cp = Cp*linmodel.A;

pmphy = ss(Ap, Bp, Cp, pm.D)
Bp(1)
Bp(2)

