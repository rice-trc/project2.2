clc
clear all
close all
addpath('../src/pnlss/')

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 
set(0,'defaultAxesFontSize',13)

%% Load Data
addnoise = false;
savefig = false;

la = load('./TRANSIENT/famp001/CLCLEF_MULTISINE.mat','t','u','y','fdof','f1','f2','df','freqs','fsamp');
la.y = la.y(:,:,:,la.fdof);

Shaker = 'no';
full = 0;
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

%% PNLSS SETUP
[Nt, P, R, n] = size(y);

MaxCount = 100;
lambda = 100;

m = size(u,2);
p = size(y,2);

% Data arbitrary
T1 = [0];
% T2 = [Nt];

% Data Periodic
% T1 = [(Ntp-1) ((1:R)-1)*Ntp+1];
T2 = [];

nx = [2 3];
ny = [];
whichtermsx = 'statesonly';
whichtermsy = 'empty';

if ~full
    tc = t+permute(t(2)*size(t,1)*(0:13), [1, 3, 2]); tc = tc(:);
else
    tc = t(:);
end
uc = u(:);
yc = y(:);

figure(1)
clf()
plot(tc, uc, '.-')
xlabel('Time (s)')
ylabel('Excitation (N)')
% print('./FIGURES/PLL_EXCITATION.eps', '-depsc')

%% How good already identified PNLSS Models are
fdirs = {'famp001','famp01','famp05','famp08','famp20'};
famps = [0.01, 0.1, 0.5, 0.8, 2.0];
fia = 5;
dat=load(sprintf('./TRANSIENT/%s/CLCLEF_MULTISINE.mat',fdirs{fia}), 'y');
load(sprintf('./Data/pnlssseqmodel_%s_nx%s.mat',fdirs{fia},sprintf('%d',nx)), 'model');
modelguess = model;
modelguess.T1 = T1;
modelguess.T2 = T2;
modelguess.E = modelguess.E;

% simulate
y_guess = fFilterNLSS(modelguess, uc);
err_linest = yc-y_guess;

figure(2)
clf()
h = [];
hold on
yd = dat.y(:);
h=[h; histogram(log10(abs(yd(abs(yd)~=0))), 'BinMethod', 'scott', 'Normalization', 'pdf')];
[~,mi]=max(h(end).Values);
modedat = h(end).BinEdges(mi)+h(end).BinWidth/2;
plot(modedat*[1 1], [0 0.7], 'k--')
h(end).EdgeColor = 'none';
xlabel('$log_{10}(|y|)$')
ylabel('Probability Density')
xlim([-10 -3])
legend(h(end), sprintf('A = %.2f N', famps(fia)))

% figure(2); legend(h(1:end), 'Location', 'northwest')
% print('./FIGURES/MSDAT_LHIST.eps', '-depsc')

figure(3)
clf()
plot(tc, yc, '-', tc, y_guess, ':'); grid on; ylim([-1 1]*2e-5); hold on;
plot(tc, ones(size(tc))*10^modedat, 'k--')
legend('Data', 'PNLSS Model', 'Location', 'northwest');
xlabel('Time t (s)')
ylabel('Displacement y')
% print(sprintf('./FIGURES/PNLSS_PLL_TRESP_%s_nx%s.eps',fdirs{fi},sprintf('%d',nx)), '-depsc')