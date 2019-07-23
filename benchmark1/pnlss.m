% estimate PNLSS model
%
%  1) estimate a nonparametric linear model from the input and 
%     noisy output data
%  2) estimate a parametric linear state-space model on the
%     nonparametric model
%  3) estimate the parameters in the full PNLSS model

close all
clearvars
% clc

srcpath = '../src/pnlss';
addpath(genpath(srcpath));
addpath('../src/matlab/')
figpath = './fig/';

%%
% addnoise = true;
addnoise = false;
savefig = true;

Alevel = 10;
% fs = 16384;
fs = 4096;
load(sprintf('data/ode45_multisine_A%d_F%d.mat',Alevel,fs))

% load('ode45A21.mat')
figure[Nt,P,R,n] = size(y);

% use middle deflection
y = PHI_L2*reshape(y,[],n)';
y = reshape(y,[Nt,P,R]);


%% Add colored noise to the output
rng(10);
if addnoise
    noise = 1e-3*std(y(:,end,end))*randn(size(y)); % Output noise signal
    % Do some filteringO
    noise(1:end-1,:,:) = noise(1:end-1,:,:) + noise(2:end,:,:);
    y = y + noise;
end

%% Separate the data in estimation, validation, and test set
% Last realization, last period for performance testing
utest = u(:,end,R); utest = utest(:);
ytest = y(:,end,R); ytest = ytest(:);

% One but last realization, last period for validation and model selection
uval = u(:,end,R-1); uval = uval(:);
yval = y(:,end,R-1); yval = yval(:);

% All other realizations for estimation. But remember to remove transient!
R = R-2;
Ptr = 3;
P = P-Ptr;
u = u(:,Ptr:end,1:R);
y = y(:,Ptr:end,1:R);

% standard deviation of the generated signal
uStd = mean(mean(std(u)));

% lines = cell2mat(cellfun(@(c) c.lines, MS, 'UniformOutput', false));
lines = MS{1}.lines;
lines(lines==1) = [];
%% Estimate nonparametric linear model (BLA)

% m: number of inputs, p: number of outputs
u = permute(u,[1,4,3,2]); % N x m x R x P
y = permute(y,[1,4,3,2]); % N x p x R x P
covY = fCovarY(y); % Noise covariance (frequency domain)

U = fft(u); U = U(lines,:,:,:); % Input spectrum at excited lines
Y = fft(y); Y = Y(lines,:,:,:); % Output spectrum at excited lines

% Estimate best linear approximation, total distortion, and noise distortion
% total and noise distortion averaged over P periods and R realizations
% total distortion level includes nonlinear and noise distortion
% G: FRF; covGML: noise + NL; covGn: noise (all only on excited lines)
[G,covGML,covGn] = fCovarFrf(U,Y); 
% figure; subplot(2,1,1); semilogy(freq(lines),abs(squeeze(G(:)))); subplot(2,1,2); plot(freq(lines),rad2deg(angle(G(:))))
%% Estimate linear state-space model (frequency domain subspace)

% Choose model order
na   = [2];
maxr = 20;
% Excited frequencies (normalized)
freq_norm = (lines-1)/Nt;

% Uncomment for uniform weighting (= no weighting)
% covGML = repmat(eye(1),[1 1 length(lines)]);

models = fLoopSubSpace(freq_norm,G,covGML,na,maxr,100);

% Extract linear state-space matrices from best model on validation data
Nval = length(uval);
tval = (0:Nval-1)/fs;
min_err = Inf;
min_na = NaN;
for n = na
    model = models{n};
    A = model{1}; B = model{2}; C = model{3}; D = model{4};
    [A,B,C] = dbalreal(A,B,C); % Compute balanced realization
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
model = models{min_na};
[A,B,C,D] = model{:};
% Balanced realization
[A,B,C] = dbalreal(A,B,C);


%% Estimate PNLSS model

% Average over periods (be careful that the data are truly steady state)
u = mean(u,4);
y = mean(y,4); 
m = 1;
p = 1;
u = u(:); % Concatenate the data: N*P*R x m
y = y(:); % N*P*R x p

% Transient settings
NTrans = 2*Nt; % Add one period before the start of each realization
% Number of transient samples and starting indices of each realization
T1 = [NTrans 1+(0:Nt:(R-1)*Nt)]; 
T2 = 0; % No non-periodic transient handling

% Nonlinear terms
nx = [3];
ny = [];
whichtermsx = 'statesonly';
% whichtermsx = 'full';
whichtermsy = 'empty';

% Settings Levenberg-Marquardt optimization
MaxCount = 200;
lambda = 100;

% Choose model order
n = min_na;

% Initial linear model in PNLSS form
model = fCreateNLSSmodel(A,B,C,D,nx,ny,T1,T2);

% Set which monomials will be optimized
model.xactive = fSelectActive(whichtermsx,n,m,n,nx);
model.yactive = fSelectActive(whichtermsy,n,m,p,ny);

% nonlinear powers
% tmp=kron([1;1],model.xpowers); tmp(model.xactive,:)

% Output of the initial linear model on the estimation data
modellinest = model;
y_lin = fFilterNLSS(modellinest,u);
errest_lin = y-y_lin;

% Compute output error on validation data
% Change transient parameter for linear model on validation data
modellinval = model;modellinval.T1 = Nt;
yval_lin = fFilterNLSS(modellinval,uval);
errval_lin = yval-yval_lin;

%Compute output error on test data
modellintest = model;modellintest.T1 = 0;
ytest_lin = fFilterNLSS(modellintest,utest);
errtest_lin = ytest-ytest_lin;

% We do not use weighting
% for kk = size(covY,3):-1:1
%     W(:,:,kk) = fSqrtInverse(covY(:,:,kk)); % Frequency weighting
% end
W = [];

% Levenberg-Marquardt optimization
[~, y_mod, models_pnlss] = fLMnlssWeighted(u,y,model,MaxCount,W,lambda);
errest_nl = y-y_mod;

% Search best model over the optimisation path on a fresh set of data
valerrs = [];
for i = 1:length(models_pnlss)
    models_pnlss(i).T1 = NTrans;
    yval_mod = fFilterNLSS(models_pnlss(i),uval); 
    valerr = yval - yval_mod;
    valerrs = [valerrs; rms(valerr)];
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

save(sprintf('./data/pnlssout_A%d_F%d.mat',Alevel,fs), 'modellinest', 'model','fs')

%% Results

% handle to figs. For saving plot
fh = {};

% optimization path
fh{1}=figure;
plot(db(valerrs));
hold on
plot(i,db(min_valerr),'r.','Markersize',10)
xlabel('Successful iteration number')
ylabel('Validation error [dB]')
title('Selection of the best model on a separate data set')

% Estimation data
plottime = [y errest_lin errest_nl];
plotfreq = fft(reshape(plottime,[Nt,R,3]));
plotfreq = squeeze(mean(plotfreq,2));

fh{2} = figure;
plot(t(1:Nt*R),plottime)
xlabel('Time (s)')
ylabel('Output (errors)')
legend('output','linear error','PNLSS error')
title('Estimation results')
disp(' ')
disp(['rms(y-y_mod) = ' num2str(rms(y-y_mod))...
    ' (= Output error of the best PNLSS model on the estimation data)'])
% disp(['rms(noise(:))/sqrt(P) = ' num2str(rms(noise(:))/sqrt(P)) ' (= Noise level)'])

fh{3} = figure;
hold on
plot(freq(1:end/2),db(plotfreq(1:end/2,:)),'.')
plot(freq(1:end/2),db(sqrt(P*squeeze(covY))), '.')
xlabel('Frequency (Hz)')
ylabel('Output (errors) (dB)')
legend('Output','Linear error','PNLSS error','noise')
title('Estimation results')

% Validation data
plottime = [yval errval_lin errval_nl];
plotfreq = fft(plottime);

fh{4} = figure;
plot(freq(1:end/2),db(plotfreq(1:end/2,:)),'.')
xlabel('Frequency (Hz)')
ylabel('Output (errors) (dB)')
legend('Output','Linear error','PNLSS error')
title('Validation results')

% Test, ie. newer seen data
plottime = [ytest errtest_lin errtest_nl];
plotfreq = fft(plottime);

fh{5} = figure;
plot(freq(1:end/2),db(plotfreq(1:end/2,:)),'.')
xlabel('Frequency');
ylabel('Output (errors) (dB)');
legend('Output','Linear error','PNLSS error')
title('Test results')

% BLA plot. We can estimate nonlinear distortion
% total and noise distortion averaged over P periods and M realizations
% total distortion level includes nonlinear and noise distortion
fh{6} = figure; hold on;
plot(freq(lines),db(G(:)))
plot(freq(lines),db(covGn(:)*R,'power'),'s')
plot(freq(lines),db(covGML(:)*R,'power'),'*')
xlabel('frequency (Hz)')
ylabel('magnitude (dB)')
title(['Estimated BLA: uStd = ' num2str(uStd)])
legend('BLA FRF','Noise Distortion','Total Distortion','Location','nw')

%% save the plots
if savefig
fstr = {'convergence','time','estim','val','test','bla_nonpar'};
figh = fh;
for i = 1:length(figh)
    fh = figh{i};
%     h_legend = findobj(fh, 'Type', 'Legend');
%     set(h_legend, 'Location', 'south')
    set(fh, 'Color', 'w');
    export_fig(fh, strcat(figpath,'pnlss_', fstr{i},'.pdf'))
end
end

