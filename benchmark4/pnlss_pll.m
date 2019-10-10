clc
clear all
close all
addpath('../src/pnlss/')

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

%% Load Data
addnoise = false;
savefig = false;

Shaker = 'no';
full = 1;
if ~full
    load(['Data/SimExp_shaker_' Shaker '_est.mat'], 't','u','y','fsamp','PNLSS');
    y = y(:,:,:,PNLSS.eval_DOF);

    val = load(['Data/SimExp_shaker_' Shaker '_est.mat'], 't','u','y','fsamp');
    val.y = val.y(:,:,:,PNLSS.eval_DOF);
else
    Npoints = 6e4;
    
    load(['Data/SimExp_full_shaker_' Shaker '_est.mat'], 't','u','y','fsamp','PNLSS');
    y = y(:,PNLSS.eval_DOF);
    
    t = t(fix(linspace(1,end,Npoints)));
    y = y(fix(linspace(1,end,Npoints)));
    u = y(fix(linspace(1,end,Npoints)));

    val = load(['Data/SimExp_full_shaker_' Shaker '_est.mat'], 't','u','y','fsamp');
    val.y = val.y(:,PNLSS.eval_DOF);    
    
    val.t = val.t(fix(linspace(1,end,Npoints)));
    val.y = val.y(fix(linspace(1,end,Npoints)));
    val.u = val.u(fix(linspace(1,end,Npoints)));
end
%% BLA
[Nt, P, R, n] = size(y);

% %% Add colored noise to the output if required
rng(10)
if addnoise
    noise = 1e-3 *std(est.y(:,end,end))*randn(size(est.y));
    
    noise(1:end-1,:,:) = noise(1:end-1,:,:) + noise(2:end,:,:);
    est.y = est.y+noise;
end

% Separate data
utest = u(:, end, end);   utest = utest(:);
ytest = y(:, end, end, :); ytest = ytest(:);

uval = val.u(:,end,:); uval = uval(:);
yval = val.y(:,end,:,:); yval = yval(:);

% Stdev
uStd = mean(mean(std(u)));

% m: number of inputs, p: number of outputs
u = permute(u, [1,4,3,2]); % N x m x R x P
y = permute(y, [1,4,3,2]); % N x p x R x P
covY = fCovarY(y);  % Noise covariance (frequency domain)

lines = 1:floor(Nt/2);
U = fft(u);  U = U(lines, :, :, :);
Y = fft(y);  Y = Y(lines, :, :, :);

[G, covGML, covGn] = fCovarFrf(U, Y);

na = 3;
maxr = 20;
freqs_norm = (lines-1)/Nt;

models = fLoopSubSpace(freqs_norm, G, covGML, na, maxr, 100);

% No validation
model = models{3};
[A,B,C,D] = model{:};
[A,B,C] = dbalreal(A,B,C);

%% PNLSS
MaxCount = 10;
lambda = 100;

m = size(u,2);
p = size(y,2);

% Data arbitrary
T1 = [0];
% T2 = [Nt];

% Data Periodic
% T1 = [Nt ((1:R)-1)*Nt+1];
T2 = [];

nx = [2 3 4 5];
ny = [];
whichtermsx = 'statesonly';
whichtermsy = 'empty';

modelguess = fCreateNLSSmodel(A,B,C,D, nx, ny, T1, T2);
modelguess.xactive = fSelectActive(whichtermsx, 3, m, 3, nx);
modelguess.xactive = fSelectActive(whichtermsx, 3, m, p, ny);
W = [];

uc = u(:);
yc = y(:);

uc_v = val.u(:);
yc_v = val.y(:);

% simulate
y_linest = fFilterNLSS(modelguess, uc);
err_linest = yc-y_linest;

y_linval = fFilterNLSS(modelguess, uc_v);
err_linval = yc_v-y_linval;

% [~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, modelguess, MaxCount, W, lambda);

modelguess.x0active = (1:modelguess.n)';
modelguess.u0active = (1:modelguess.m)';
[~, y_mod, models_pnlss] = fLMnlssWeighted_x0u0(uc, yc, modelguess, MaxCount, W, lambda);

err_nlest = yc-y_mod;