clc
clear all
close all
addpath('../src/pnlss/')

%% Combined data
fchar = 'comb';
fdir = 'famp01';
f01 = load(sprintf('./TRANSIENT/%s/CLCLEF_MULTISINE.mat',fdir), ...
    'u', 'y', 'fdof', 't', 'f1', 'f2', 'df', 'freqs', 'fsamp');
fdir = 'famp05';
f05 = load(sprintf('./TRANSIENT/%s/CLCLEF_MULTISINE.mat',fdir), ...
    'u', 'y', 'fdof', 't', 'f1', 'f2', 'df', 'freqs', 'fsamp');

y = cat(3, f01.y, f05.y);
u = cat(3, f01.u, f05.u);
fdof = f01.fdof;
t = f01.t;
f1 = f01.f1;
f2 = f01.f2;
df = f01.df;
freqs = f01.freqs;
fsamp = f01.fsamp;
Rlsns = size(u,3);

%%
[Nt, P, R, n] = size(y);

% Use excitation node response
y = y(:, :, :, fdof);

% Separate data
utest = u(:, end, end);   utest = utest(:);
ytest = y(:, end, end, :); ytest = ytest(:);

uval = u(:,end,R-1); uval = uval(:);
yval = y(:,end,R-1,:); yval = yval(:);

% All other repeats for estimation
R = R-2;
Ptr = 6;
P = P-Ptr;
u = u(:, Ptr:end, 1:R);
y = y(:, Ptr:end, 1:R, :);

% Standard deviation of generated signal
uStd = mean(mean(std(u)));

% Non-parametric BLA model

% m: number of inputs, p: number of outputs
u = permute(u, [1,4,3,2]); % N x m x R x P
y = permute(y, [1,4,3,2]); % N x p x R x P
covY = fCovarY(y);  % Noise covariance (frequency domain)

lines = (f1/df+1):(f2/df+1);
U = fft(u);  U = U(lines, :, :, :);  % Input Spectrum at excited lines
Y = fft(y);  Y = Y(lines, :, :, :);  % Output Spectrum at excited lines

[G, covGML, covGn] = fCovarFrf(U, Y);

% Estimate linear state-space model (frequency domain subspace)
% Model order
na = 3;
maxr = 20;
% Excited frequencies (normed)
freqs_norm = (lines-1)/Nt;

models = fLoopSubSpace(freqs_norm, G, covGML, na, maxr, 100);

% Extract linear state-space matrices from best model on validation data
Nval = length(uval);
tval = (0:Nval-1)/fsamp;
min_e = Inf;
min_na = NaN;
for n = na
    model = models{n};
    A = model{1}; B = model{2}; C = model{3}; D = model{4};
    [A,B,C] = dbalreal(A,B,C);  % Balance realizations
    yval_hat = lsim(ss(A,B,C,D,1/fsamp), uval, tval);
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
[A,B,C] = dbalreal(A,B,C);

%% Estimate PNLSS Model

u = mean(u, 4);
y = mean(y, 4);
m = size(u,2);
p = size(y,2);
uc = u(:);
yc = y(:);

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
y_lin = fFilterNLSS(modellinest, uc);
errest_lin = yc-y_lin;

modellinval = model;  modellinval.T1 = Nt;
yval_lin = fFilterNLSS(modellinval, uval);
errval_lin = yval-yval_lin;

modellintest = model; modellintest.T1 = 0;
ytest_lin = fFilterNLSS(modellintest, utest);
errtest_lin = ytest-ytest_lin;

W = [];

[~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, model, MaxCount, W, lambda);
errest_nl = yc-y_mod;

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

