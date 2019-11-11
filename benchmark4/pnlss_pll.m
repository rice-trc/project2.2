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

la = load('./TRANSIENT/famp001/CLCLEF_MULTISINE.mat','t','u','y','fdof','f1','f2','df','freqs','fsamp');
la.y = la.y(:,:,:,la.fdof);

Shaker = 'no';
full = 1;
if ~full
    load(['Data/SimExp_shaker_' Shaker '_est.mat'], 't','u','y','fsamp','PNLSS');
    y = y(:,:,:,PNLSS.eval_DOF);

    val = load(['Data/SimExp_shaker_' Shaker '_est.mat'], 't','u','y','fsamp');
    val.y = val.y(:,:,:,PNLSS.eval_DOF);
    
    % Undersampling
    fsamp = fsamp/10;
    y = y(1:10:end,:,:);
    u = u(1:10:end,:,:);
    t = t(1:10:end,:,:);
    
    val.fsamp = val.fsamp/10;
    val.y = val.y(1:10:end,:,:);
    val.u = val.u(1:10:end,:,:);
    val.t = val.t(1:10:end,:,:);
else    
    load(['Data/SimExp_full_shaker_' Shaker '_est.mat'], 't','u','y','fsamp','PNLSS');
    y = y(:,PNLSS.eval_DOF);
    
	% Undersampling
    fsamp = la.fsamp;
    y = interp1(t, y, (0:1/fsamp:t(end))');
    u = interp1(t, u, (0:1/fsamp:t(end))');
    t = (0:1/fsamp:t(end))';
    
    val = load(['Data/SimExp_full_shaker_' Shaker '_est.mat'], 't','u','y','fsamp');
    val.y = val.y(:,PNLSS.eval_DOF);    
    
    val.fsamp = la.fsamp;
    val.y = interp1(val.t, val.y, (0:1/fsamp:val.t(end))');
    val.u = interp1(val.t, val.u, (0:1/fsamp:val.t(end))');
    val.t = (0:1/fsamp:val.t(end))';
end

fdirs = {'famp001','famp01','famp05','famp08','famp20'};
famps = [0.01, 0.1, 0.5, 0.8, 2.0];

%%
fia = 1;
for fia=1:length(fdirs)
% %% PNLSS
[Nt, P, R, n] = size(y);

MaxCount = 100;
lambda = 100;

m = size(u,2);
p = size(y,2);

% Data arbitrary
T1 = [0];
% T2 = [Nt];

% Data Periodic
% T1 = [(Nt-1)/4 ((1:R)-1)*Nt+1];
T2 = [];

nx = [2 3];
ny = [];
whichtermsx = 'statesonly';
whichtermsy = 'empty';

load(sprintf('./Data/pnlssseqmodel_%s_nx%s.mat',fdirs{fia},sprintf('%d',nx)), 'model');
modelguess = model;
modelguess.T1 = T1;
modelguess.T2 = T2;
modelguess.xactive = fSelectActive(whichtermsx, 3, m, 3, nx);
modelguess.xactive = fSelectActive(whichtermsx, 3, m, p, ny);
W = [];

% INITIAL GUESS ON E
% modelguess.E = rand(size(modelguess.E))*0;
% ---------

if ~full
    tc = t+permute(t(2)*size(t,1)*(0:13), [1, 3, 2]); tc = tc(:);
else
    tc = t(:);
end
uc = u(:);
yc = y(:);

uc_v = val.u(:);
yc_v = val.y(:);

% simulate
y_linest = fFilterNLSS(modelguess, uc);
err_linest = yc-y_linest;

y_linval = fFilterNLSS(modelguess, uc_v);
err_linval = yc_v-y_linval;
me=[];
try
    [~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, modelguess, MaxCount, W, lambda);

    % modelguess.x0active = (1:modelguess.n)';
    % modelguess.u0active = (1:modelguess.m)';
    % [~, y_mod, models_pnlss] = fLMnlssWeighted_x0u0(uc, yc, modelguess, MaxCount, W, lambda);
catch me
    modelguess.E = rand(size(modelguess.E))*0.1;
    
    [~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, modelguess, MaxCount, W, lambda);
end

% Choose best model from PNLSS (using validation data)
valerrs = [];
for i=1:length(models_pnlss)        
    yval_mod = fFilterNLSS(models_pnlss(i), uc_v);
	valerr = yc_v - yval_mod;
	valerrs = [valerrs; rms(valerr)];
end

[min_valerr, i] = min(valerrs);
model = models_pnlss(i);

y_mod = fFilterNLSS(model, uc);
err_nlest = yc-y_mod;

save(sprintf('./Data/pnlss_pll_%s_nx%s.mat', fdirs{fia}, sprintf('%d',nx)), 'model', 'fsamp');
% %%
figure(10); clf()
plot(tc, yc); hold on
plot(tc, y_mod, ':'); hold on
if isempty(me)
    plot(tc, y_linest, ':');
end
xlabel('Time (s)')
ylabel('Displacement y')
ylim([-1 1]*2e-5)

print(sprintf('./FIGURES/TDOMPERF_PNLSS_PLL_%s_nx%s.eps',fdirs{fia},sprintf('%d',nx)), '-depsc');
% plot(err_linest)
% plot(tc, err_nlest)
end

%% Assess Linear eigenvalues
Wseq = zeros(length(fdirs),1);
Wpll = zeros(length(fdirs),1);
for fia=1:length(fdirs)
    nx = [2 3];
    
    seq = load(sprintf('./Data/pnlssseqmodel_%s_nx%s.mat',fdirs{fia},sprintf('%d',nx)), 'model');
    pll = load(sprintf('./Data/pnlss_pll_%s_nx%s.mat', fdirs{fia}, sprintf('%d',nx)), 'model');
    
    Pseq = pole(d2c(ss(seq.model.A, seq.model.B, seq.model.C, seq.model.D, 1./fsamp)));
    Ppll = pole(d2c(ss(pll.model.A, pll.model.B, pll.model.C, pll.model.D, 1./fsamp)));
    
    [~,sseq] = sort(imag(Pseq));
    [~,spll] = sort(imag(Ppll));
%     disp(table(Pseq(sseq([1 end])), Ppll(spll([1 end]))))
    
    Wseq(fia) = imag(Pseq(sseq(end)));
    Wpll(fia) = imag(Ppll(spll(end)));
end
table(fdirs',Wseq/2/pi,Wpll/2/pi)