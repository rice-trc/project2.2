%clc
clear variables
%close all
addpath('~/src/matlab/pnlss/')
addpath('~/src/matlab/misc/')

include_vel = true;
%include_vel = false;

add_noise = true;
%% Generate multisine input (u)
rng('default')
RMSu = 700; % Root mean square value for the input signal
N = 8192; %2048;    % Number of samples
R = 2;       % Number of phase realizations (one for validation and one for performance testing)
P = 3;       % Number of periods
kind = 'full';           % 'Full','Odd','SpecialOdd', or 'RandomOdd': kind of multisine
M = round(0.2*N/2);     % Last excited line
[u,lines,non_exc_odd,non_exc_even] = fMultisine(N, kind, M, R); % Multisine signal, excited and detection lines
u = u/rms(u(:,1))*RMSu; % Scale multisine to the correct rms level
u = repmat(u,[1 1 P]);  % N x R x P
u = permute(u,[1 3 2]); % N x P x R

%% generate data
rng('default')
fMin = 5;
fMax = 100;
fs = 700;
options.N = N;
options.P = P;
options.M = R;
options.fMin = fMin;   % Hz
options.fMax = fMax;   % Hz
options.fs = fs;       % Hz
options.type = 'full'; %'full', 'odd', 'oddrandom'
options.std = RMSu;
[u, lines] = fMultiSinGen(options);
u = reshape(u, [N, P, R]);

% lines = lines(1:200);

%% state space model
load('data/system.mat')
% Continuous time model
ctmodel.A = [zeros(size(M)) eye(size(M));
            -M\K -M\C];
ctmodel.B = [zeros(size(M)); inv(M)];
ctmodel.C = [eye(size(M)) zeros(size(M))];
ctmodel.D = zeros(size(M));

% include velocity
if include_vel == true
ctmodel.C = [ctmodel.C; zeros(size(M)) eye(size(M))];
ctmodel.D = [ctmodel.D; zeros(size(M))];
end

dtm = c2d(ss(ctmodel.A,ctmodel.B,ctmodel.C,ctmodel.D),1/fs, 'tustin');

p = size(ctmodel.C,1);
nx = 0;
ny = 0;
T1 =  [N 1+(0:P*N:(R-1)*P*N)];
T2 = 0;
true_model = fCreateNLSSmodel(dtm.A,dtm.B,dtm.C,dtm.D,nx,ny,T1,T2);
uext = zeros(N*P*R, 3);
uext(:,1) = u(:);
y = fFilterNLSS(true_model,uext);
y = reshape(y,[N P R, p]); % N x P x R x p
if include_vel == true
    y = y(:,:,:, [1,2,3,6]);
end

if add_noise
noise = 1e-3*std(y(:,end,end))*randn(size(y)); % Output noise signal
noise(1:end-1,:,:) = noise(1:end-1,:,:) + noise(2:end,:,:); % Do some filtering
y = y + noise; % Noise added to the output
end

save('data/pnlss.mat','y','u','lines','fs')

%% load data from python

%load('data/ms_A700_upsamp5_fs750_eps0.mat')
% % convert from python to matlab format
%lines = double(lines + 1);
%fs = double(fs);
%u = permute(u, [1,4,3,2]);
%y = permute(y, [1,4,3,2]);

%%
[N, P, R, p] = size(y);
[N, P, R, m] = size(u);

% Last realization, last period for performance testing
utest = u(:,end,R,:); utest = reshape(utest,[],m);
ytest = y(:,end,R,:); ytest = reshape(ytest,[],p);

% One but last realization, last period for validation and model selection
uval = u(:,end,R-1,:); uval = reshape(uval,[],m);
yval = y(:,end,R-1,:); yval = reshape(yval,[],p);

% All other repeats for estimation\
Ptr = 1;
uest = u(:,Ptr+1:end,1:R,:);
yest = y(:,Ptr+1:end,1:R,:);

% standard deviation of the generated signal
uStd = mean(mean(std(uest)));

%% Estimate nonparametric linear model (BLA)

time = 0:1/fs:N*P/fs - 1/fs;
freq = 0:fs/N:fs-fs/N;

% m: number of inputs, p: number of outputs
uest = permute(uest,[1,4,3,2]); % N x m x R x P
yest = permute(yest,[1,4,3,2]); % N x p x R x P
covY = fCovarY(yest); % Noise covariance (frequency domain)

U = fft(uest);  U = U(lines, :, :, :);  % Input Spectrum at excited lines
Y = fft(yest);  Y = Y(lines, :, :, :);  % Output Spectrum at excited lines

[G, covGML, covGn] = fCovarFrf(U, Y);

figure; subplot(2,1,1); hold on; plot(freq(lines), squeeze(db(abs(G(1,1,:)))))
plot(freq(lines), squeeze(db(abs([covGML(1,1,:)*R covGn(1,1,:)*R]),'power')),'.')
xlabel('Frequency (Hz)'); ylabel('Amplitude (dB)')
legend('FRF','Total distortion','Noise distortion')
subplot(2,1,2); plot(freq(lines),rad2deg(angle(squeeze(G(1,1,:)))))
xlabel('Frequency (Hz)'); ylabel('Angle (degree)')

%% Estimate linear state-space model (frequency domain subspace)
% Model order
na = [6];
maxr = 20;
% Excited frequencies (normed)
freqs_norm = (lines-1)/N;

models = fLoopSubSpace(freqs_norm, G, 0, na, maxr, 100);
models = fLoopSubSpace(freqs_norm, G, covGML, na, maxr, 100);

% Extract linear state-space matrices from best model on validation data
Nval = length(uval);
tval = (0:Nval-1)/fs;
min_e = Inf;
min_na = NaN;
for n = na
    model = models{n};
    A = model{1}; B = model{2}; C = model{3}; D = model{4};
    %[A,B,C] = dbalreal(A,B,C);  % Balance realizations
    yval_hat = lsim(ss(A,B,C,D,1/fs), uval, tval);
    % If with error
%     Rms value of last period of error
%     err = sqrt(mean(err(end-Nt+1:end).^2));
%     if err < min_err
        min_na = n;
%         min_err = err;
%     end  
end
min_na = na(1);
% Select the best model
model = models{min_na};
[A,B,C,D] = model{:};

% [A,B,C] = dbalreal(A,B,C);

return
%% Estimate PNLSS Model

% Average over periods (be careful that the data are truly steady state)
uest = mean(uest,4); 
yest = mean(yest,4);  % N x p x R
uest = permute(uest,[1,3,2]); % N x R x m
yest = permute(yest,[1,3,2]); % N x R x p
uest = reshape(uest,Nt*R,m); % Concatenate the data: N*P*R x m
yest = reshape(yest,Nt*R,p); % N*P*R x p

% Transient settings
Ntrans = Nt;
T1 = [Ntrans 1+(0:Nt:(R-1)*Nt)];
T2 = 0;

nx = [2 3];
ny = [2 3];
whichtermsx = 'full';
whichtermsy = 'full';

% Settings for LM opt
MaxCount = 100;
lambda = 100;

n = min_na;

model = fCreateNLSSmodel(A, B, C, D, nx, ny, T1, T2);

model.xactive = fSelectActive(whichtermsx, n, m, n, nx);
model.yactive = fSelectActive(whichtermsy, n, m, p, ny);

modellinest = model;
y_lin = fFilterNLSS(modellinest, uest);
errest_lin = yest-y_lin;

modellinval = model;  modellinval.T1 = Nt;
yval_lin = fFilterNLSS(modellinval, uval);
errval_lin = yval-yval_lin;

modellintest = model; modellintest.T1 = 0;
ytest_lin = fFilterNLSS(modellintest, utest);
errtest_lin = ytest-ytest_lin;

W = [];

[~, y_mod, models_pnlss] = fLMnlssWeighted(uest, yest, model, MaxCount, W, lambda);
errest_nl = yest-y_mod;

valerrs = [];
for i=1:length(models_pnlss)
    models_pnlss(i).T1 = Ntrans;
    yval_mod = fFilterNLSS(models_pnlss(i), uval);
    valerr = yval - yval_mod;
    valerrs = [valerrs; rms(valerr)];
end

[min_valerr,i] = min(valerrs);
model = models_pnlss(i);

% Compute output error on validation data
modelval = model;modelval.T1 = Ntrans;
yval_nl = fFilterNLSS(modelval,uval);
errval_nl = yval-yval_nl;
% Compute output error on test data
modeltest = model;modeltest.T1 = 0;
ytest_nl = fFilterNLSS(modeltest,utest);
errtest_nl = ytest-ytest_nl;

err = [rms(errest_lin), rms(errest_nl); rms(errval_lin), rms(errval_nl);...
    rms(errtest_lin), rms(errtest_nl)];

fprintf('############# RMS errors #############\n')
fprintf('e_est_lin:\t %0.3e\t e_est_nl:\t %0.3e\n', err(1,:))
fprintf('e_val_lin:\t %0.3e\t e_val_nl:\t %0.3e\n', err(2,:))
fprintf('e_test_lin:\t %0.3e\t e_test_nl:\t %0.3e\n',err(3,:))

save(sprintf('./pnlss%s.mat',fchar), 'modellinest', 'model')

%% Results

% handle to figs. For saving plot
fh = {};

fh{1}=figure;
plot(db(valerrs));
hold on
plot(i,db(min_valerr),'r.','Markersize',10)
xlabel('Successful iteration number')
ylabel('Validation error [dB]')
title('Selection of the best model on a separate data set')

% Estimation data
plottime = [yc errest_lin errest_nl];
plotfreq = fft(reshape(plottime,[Nt,R,3]));
plotfreq = squeeze(mean(plotfreq,2));
pause(0.1)


fh{2} = figure;
plot(t(1:Nt*R),plottime)
xlabel('Time (s)')
ylabel('Output (errors)')
legend('output','linear error','PNLSS error')
title('Estimation results')
disp(' ')
disp(['rms(y-y_mod) = ' num2str(rms(yc-y_mod))...
    ' (= Output error of the best PNLSS model on the estimation data)'])
% disp(['rms(noise(:))/sqrt(P) = ' num2str(rms(noise(:))/sqrt(P)) ' (= Noise level)'])
print(sprintf('./FIGURES/PNLSS_%s_TDOMRES.eps',fchar), '-depsc')
pause(0.1)

freq = (0:Nt-1)/Nt*fsamp;

fh{3} = figure;
hold on
plot(freq(1:end/2),db(plotfreq(1:end/2,:)),'.')
plot(freq(1:end/2)/2,db(sqrt(P*squeeze(covY))), '.')
xlabel('Frequency (Hz)')
ylabel('Output (errors) (dB)')
legend('Output','Linear error','PNLSS error','noise')
title('Estimation results')

% Validation data
plottime = [yval errval_lin errval_nl];
plotfreq = fft(plottime);
pause(0.1)

fh{4} = figure;
plot(freq(1:end/2),db(plotfreq(1:end/2,:)),'.')
xlabel('Frequency (Hz)')
ylabel('Output (errors) (dB)')
legend('Output','Linear error','PNLSS error')
title('Validation results')
print(sprintf('./FIGURES/PNLSS_%s_VALIDAT.eps', fchar), '-depsc')

% Test, ie. newer seen data
plottime = [ytest errtest_lin errtest_nl];
plotfreq = fft(plottime);
pause(0.1)

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
print(sprintf('./FIGURES/PNLSS_%s_BLA_distort.eps', fchar), '-depsc')
pause(0.1)

% %% save the plots
% if savefig
% fstr = {'convergence','time','estim','val','test','bla_nonpar'};
% figh = fh;
% for i = 1:length(figh)
%     fh = figh{i};
% %     h_legend = findobj(fh, 'Type', 'Legend');
% %     set(h_legend, 'Location', 'south')
%     set(fh, 'Color', 'w');
%     export_fig(fh, strcat(figpath,'pnlss_', fstr{i},'.pdf'))
% end
% end

