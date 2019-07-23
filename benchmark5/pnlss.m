% estimate PNLSS model
%
%  1) estimate a nonparametric linear model from the input and 
%     noisy output data
%  2) estimate a parametric linear state-space model on the
%     nonparametric model
%  3) estimate the parameters in the full PNLSS model

% close all
clear variables
% clearvars  % does not wotk in octave
% clc

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
srcdir = '../src/pnlss';
addpath(srcdir);
srcdir = '../src/matlab';
addpath(genpath(srcdir));

savename = 'pnlss1';
benchmark = 2;
data.A = 10;
data.name = 'up8_ms_full';
% data.name = 'ode8_test';

show_ms = false;
show_pnlss = true;
%%
% addnoise = true;
addnoise = false;
savefig = true;

% ensure it is possible to show the display. Octave compatible
% https://stackoverflow.com/a/30240946
scr = get(0,'ScreenSize');
if isequal(scr(3:4),[1 1])
    show_ms = false;
    show_pnlss = false;
end

if isOctave
    pkg load control % for ss/lsim
    pkg load signal  % for rms
end

load(sprintf('data/b%d_A%d_%s',benchmark,data.A,data.name));
if show_ms
    % plot the middle deflection
    phi = sys.PHI([sys.L/2]);
    ms_plot(t,y,u,freq,MS{1}.lines,phi,benchmark,data.A,data.name) %savefig
end

% tmp=load(sprintf('data/b%d_A%d_%s',benchmark,data.A,data.name));
% tmp.PHIS = PHIS;
% tmp.sys = [];
% save(sprintf('data/b%d_A%d_%s_modif',benchmark,data.A,data.name),'-struct','tmp')

[Nt,P,R,p] = size(y);
[Nt,P,R,m] = size(u);

% create measurement points
switch benchmark
    case 1
        PHIS = sys.PHI([sys.L/2]);
        % Nonlinear terms
        nx = [3];
        ny = [];
        whichtermsx = 'statesonly'; % full
        whichtermsy = 'empty';
    case 2
        PHIS = sys.PHI([sys.L/4; sys.L/2; 3/4*sys.L]);
        nx = [3];
        ny = [];
        whichtermsx = 'statesonly'; % full
        whichtermsy = 'empty';
    case 3
        PHIS = sys.PHI([sys.L/4; sys.L/2; 3/4*sys.L]);
        nx = [2,3];
        ny = [];
        whichtermsx = 'statesonly'; % full
        whichtermsy = 'empty';
end

% convert to measurement points using modal shapes from Phi.
nPHIS = size(PHIS,1);
yres = zeros(Nt,P,R,nPHIS); 
for i = 1:nPHIS
    yres(:,:,:,i) = reshape(PHIS(i,:)*reshape(y,[],p)', [Nt,P,R,1]);
end
y = yres;
p = nPHIS;

figure;
r = 1;
per = (y(:,1:end-1,r)-y(:,end,r)) / rms(y(:,1,r));
plot(t(1:Nt*(P-1)),db(per(:)),'k-')
% indicate periods
h1 = vline(t((1:r*(P-1))*Nt),'--g');
% indicate realizations
h2 = vline(t((1:r)*Nt*(P-1)),'--k');set([h1 h2],'LineWidth',0.5)
xlabel('time (s)')
ylabel('Relative error to last period (dB)')
title([num2str(Nt) ' samples per period'])


%% Add colored noise to the output
if addnoise
    rng(10);
    noise = 1e-3*std(y(:,end,end))*randn(size(y)); % Output noise signal
    noise1 = noise;
    % Do some filtering
    noise(1:end-1,:,:) = noise(1:end-1,:,:) + noise(2:end,:,:);
    y = y + noise;
    
    dft1 = fft(noise1(:,1,1)); dft2 = fft(noise(:,1,1));
    figure; hold on
    plot(abs(dft1(1:end/2)),'-k');
    plot(abs(dft2(1:end/2)),'Color',[0 0 1 0.3])
    ylabel('Magnitude (dB)')
    xlabel('Frequency line')
    legend('Flat spectrum','filtered')
end

%% Separate the data in estimation, validation, and test set
% Last realization, last period for performance testing
utest = u(:,end,R,:); utest = reshape(utest,[],m);
ytest = y(:,end,R,:); ytest = reshape(ytest,[],p);

% One but last realization, last period for validation and model selection
uval = u(:,end,R-1,:); uval = reshape(uval,[],m);
yval = y(:,end,R-1,:); yval = reshape(yval,[],p);

% All other realizations for estimation. But remember to remove transient!
R = R-2;
Ptr = 4;
P = P-Ptr;
uest = u(:,Ptr:end,1:R,:);
yest = y(:,Ptr:end,1:R,:);

% standard deviation of the generated signal
uStd = mean(mean(std(uest)));

%% Estimate nonparametric linear model (BLA)

% m: number of inputs, p: number of outputs
uest = permute(uest,[1,4,3,2]); % N x m x R x P
yest = permute(yest,[1,4,3,2]); % N x p x R x P
covY = fCovarY(yest); % Noise covariance (frequency domain)


%%
lines = MS{1}.lines;

U = fft(uest); U = U(lines,:,:,:); % Input spectrum at excited lines
Y = fft(yest); Y = Y(lines,:,:,:); % Output spectrum at excited lines

% Estimate best linear approximation, total distortion, and noise distortion
% total and noise distortion averaged over P periods and R realizations
% total distortion level includes nonlinear and noise distortion
% G: FRF; covGML: noise + NL; covGn: noise (all only on excited lines)
[G,covGML,covGn] = fCovarFrf(U,Y); 
bla.G = G; bla.covGML = covGML; bla.covGn = covGn; bla.freq = freq;
bla.lines = lines; bla.covY = covY; bla.fs = fs;
% figure; subplot(2,1,1); hold on; plot(freq(lines), db(abs(G(:))))
% plot(freq(lines), db(abs([covGML(:)*R covGn(:)*R]),'power'),'.')
% xlabel('Frequency (Hz)'); ylabel('Amplitude (dB)')
% legend('FRF','Total distortion','Noise distortion')
% subplot(2,1,2); plot(freq(lines),rad2deg(angle(G(:))))
% xlabel('Frequency (Hz)'); ylabel('Angle (degree)')


%% Estimate linear state-space model (frequency domain subspace)

% Choose model order
na   = [2,3]; % [2,3,4]
maxr = 20;%20;
% Excited frequencies (normalized). -1 because lines are given wrt. fft.
freq_norm = (bla.lines-1)/Nt;
% alternative: freq(lines)/fs
% Uncomment for uniform weighting (= no weighting)
% covGML = repmat(eye(1),[1 1 length(lines)]);
%% NBNBNB. NO WEIGTHING. ADD AGAIN! covGML
[linmodels, subspacedata, ~]= fLoopSubSpace(freq_norm,bla.G,0,na,maxr,100);

% Extract linear state-space matrices from best model on validation data
Nval = length(uval);
tval = (0:Nval-1)/fs;
min_err = Inf;
min_na = NaN;
for n = na
    model = linmodels{n};
    A = model{1}; B = model{2}; C = model{3}; D = model{4};
    if ~isOctave
        [A,B,C] = dbalreal(A,B,C); % Compute balanced realization
    end
    yval_hat = lsim(ss(A,B,C,D,1/fs),uval,tval);
    err = yval - yval_hat; 
    % Rms value of the last period of the error signal
    err = sqrt(mean(err(end-Nt+1:end).^2));
    if err < min_err
        min_na = n;
        min_err = err;
    end
end
% select the best model
min_na = 2;
model = linmodels{min_na};
[A,B,C,D] = model{:};
if ~isOctave
    % Balanced realization
    [A,B,C] = dbalreal(A,B,C);
end
n = length(A);
fprintf('Model order selected, n: %d\n',n)

% plot subspace models
hfig = fPlotSubSpace(subspacedata,linmodels,bla.G,bla.covGML,bla.freq(bla.lines),fs,na,maxr);
h = hfig{1}{1};
title(h.CurrentAxes,'Subspace models. Stars: LM optimized')

% return;

%% Estimate PNLSS model

% Average over periods (be careful that the data are truly steady state)
uest = mean(uest,4); 
yest = mean(yest,4);  % N x p x R
uest = permute(uest,[1,3,2]); % N x R x m
yest = permute(yest,[1,3,2]); % N x R x p
uest = reshape(uest,Nt*R,m); % Concatenate the data: N*P*R x m
yest = reshape(yest,Nt*R,p); % N*P*R x p

% Transient settings
NTrans = 2*Nt; % Add one period before the start of each realization
% Number of transient samples and starting indices of each realization
T1 = [NTrans 1+(0:Nt:(R-1)*Nt)]; 
T2 = 0; % No non-periodic transient handling

% Settings Levenberg-Marquardt optimization
MaxCount = 100;
lambda = 100;

% Initial linear model in PNLSS form
model = fCreateNLSSmodel(A,B,C,D,nx,ny,T1,T2);

% Set which monomials will be optimized
model.xactive = fSelectActive(whichtermsx,n,m,n,nx);
model.yactive = fSelectActive(whichtermsy,n,m,p,ny);

% nonlinear powers
% tmp=kron([1;1],model.xpowers); tmp(model.xactive,:)

% Output of the initial linear model on the estimation data
modellinest = model;
y_lin = fFilterNLSS(modellinest,uest);
errest_lin = yest-y_lin;

% Compute output error on validation data
% Change transient parameter for linear model on validation data
modellinval = model;modellinval.T1 = Nt;
yval_lin = fFilterNLSS(modellinval,uval);
errval_lin = yval-yval_lin;

%Compute output error on test data
modellintest = model;modellintest.T1 = 0;
ytest_lin = fFilterNLSS(modellintest,utest);
errtest_lin = ytest-ytest_lin;

% We do not use weighting for PNLSS models.
% for kk = size(covY,3):-1:1
%     W(:,:,kk) = fSqrtInverse(covY(:,:,kk)); % Frequency weighting
% end
W = [];

% Levenberg-Marquardt optimization
[~, y_mod, models_pnlss] = fLMnlssWeighted(uest,yest,model,MaxCount,W,lambda);
errest_nl = yest-y_mod;

% Search best model over the optimisation path on a fresh set of data
valerrs = zeros(length(models_pnlss),1);
valerr = cell(length(models_pnlss),1);
for i = 1:length(models_pnlss)
    models_pnlss(i).T1 = NTrans;
    yval_mod = fFilterNLSS(models_pnlss(i),uval); 
    valerr{i} = rms(yval - yval_mod,1);
    valerrs(i) = sum(valerr{i});
end

% Select the best model on the validation data to avoid overfitting
[min_valerr,i] = min(valerrs);
model = models_pnlss(i);

% Compute output error on validation data
modelval = model;modelval.T1 = NTrans;
yval_nl = fFilterNLSS(modelval,uval);
errval_nl = yval-yval_nl;
% Compute output error on test data
modeltest = model;modeltest.T1 = 0;
ytest_nl = fFilterNLSS(modeltest,utest);
errtest_nl = ytest-ytest_nl;

% error
err = [rms(errest_lin), rms(errest_nl); rms(errval_lin), rms(errval_nl);...
    rms(errtest_lin), rms(errtest_nl)];

fprintf('############# RMS errors #############\n')
fprintf('e_est_lin:\t %0.3e\t e_est_nl:\t %0.3e\n', err(1,:))
fprintf('e_val_lin:\t %0.3e\t e_val_nl:\t %0.3e\n', err(2,:))
fprintf('e_test_lin:\t %0.3e\t e_test_nl:\t %0.3e\n',err(3,:))

estdata = [yest errest_lin errest_nl];
valdata = [yval errval_lin errval_nl];
testdata = [ytest errtest_lin errtest_nl];

sig = struct('P',P, 'R',R, 'Nt',Nt, 'p',p, 'm',m, 'fs',fs);
save(sprintf('data/b%d_A%d_%s_%s.mat',benchmark,data.A,data.name,savename),...
    'modellinest','model','valerr','valerrs','estdata','valdata','testdata',...
    'bla','sig','linmodels','subspacedata')

%% Results

% convert to continious time for modal analysis of linear part
sys_ct = d2c(ss(model.A,model.B,model.C,model.D,1/fs));
sd = modal(sys_ct.A,sys_ct.C);
wn = sprintf('%0.5g, ', sd.wn);
zeta = sprintf('%0.5g, ', sd.zeta);
disp('Identified Modal Parameters')
fprintf('Nat freq %s Hz. \ndamping %s\n',wn,zeta)

% subspace model
sys_ct_sub = d2c(ss(modellinest.A,modellinest.B,modellinest.C,modellinest.D,1/sig.fs));
sd = modal(sys_ct_sub.A,sys_ct_sub.C);
wn = sprintf('%0.5g, ', sd.wn);
zeta = sprintf('%0.5g, ', sd.zeta);
disp('Identified Linear Modal Parameters')
fprintf('Nat freq %s Hz. \ndamping %s\n',wn,zeta)

% similarity transform to get state space matrices in physical coordinates
if p == 1 %% so far, only works for one input
    [Ap,Bp,Cp,T] = ss2phys(sys_ct.A,sys_ct.B,sys_ct.C);
    sys_phys = ss(Ap,Bp,Cp,model.D);
end
%% plot
if show_pnlss
    figname = sprintf('b%d_A%d_%s',benchmark,data.A,data.name );
    pnlss_plot(t,sig,bla,estdata,valdata,testdata,valerrs,model,hfig,savefig,figname)
end