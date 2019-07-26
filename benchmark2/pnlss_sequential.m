clc
clear all

srcpath = '../src/pnlss';
addpath(genpath(srcpath));
addpath('../src/matlab/')
figpath = './fig';

%% 
addnoise = false;
savefig  = true;

Alevels = [0.01 0.05 0.10 0.15 0.20 0.25];

mds = 123;

PerOrNot = [true true true true true true true true true true true];  % true if periodic; false if not
% fs = 4096;
fs = 16384;

%% BLA with lowest forcing
load(sprintf('./data/ode45_multisine_A%.2f_F%d_mds%d.mat',Alevels(1), fs, mds))

freq = (0:Nt-1)*f0;
[Nt, P, R, n] = size(y);

% Use middle deflection
% y = PHI_L2*reshape(y, [], n)';
yout = sum(y.*permute(PHI_L2, [1 3 4 2]),4);
y = yout;
y = reshape(y, [Nt, P, R]);

% %% Add colored noise to the output if required
rng(10)
if addnoise
    noise = 1e-3 *std(y(:,end,end))*randn(size(y));
    
    noise(1:end-1,:,:) = noise(1:end-1,:,:) + noise(2:end,:,:);
    y = y+noise;
end

% Separate the data in estimation, validation and test set
utest = u(:, end, end);  utest = utest(:);
ytest = y(:, end, end);  ytest = ytest(:);

uval = u(:, end, R-1);  uval = uval(:);
yval = y(:, end, R-1);  yval = yval(:);

% All other repeats for estimation
R = R-2;
Ptr = 6;
P = P-Ptr;
u = u(:, Ptr:end, 1:R);
y = y(:, Ptr:end, 1:R);

% Standard deviation of generated signal
uStd = mean(mean(std(u)));

% Non-parametric BLA model

% m: number of inputs, p: number of outputs
u = permute(u, [1,4,3,2]); % N x m x R x P
y = permute(y, [1,4,3,2]); % N x p x R x P
covY = fCovarY(y);  % Noise covariance (frequency domain)

lines = MS{1}.lines;
lines(lines==1) = [];
U = fft(u);  U = U(lines, :, :, :);  % Input Spectrum at excited lines
Y = fft(y);  Y = Y(lines, :, :, :);  % Output Spectrum at excited lines

[G, covGML, covGn] = fCovarFrf(U, Y);

% Estimate linear state-space model (frequency domain subspace)
% Model order
if mds>1
    na = [2, 3, 4, 5, 6, 7, 8];
else
    na = [2];
end
maxr = 20;
% Excited frequencies (normed)
freqs_norm = (lines-1)/Nt;

% covGML
models = fLoopSubSpace(freqs_norm, G, covGML, na, maxr, 100);

% Extract linear state-space matrices from best model on validation data
Nval = length(uval);
tval = (0:Nval-1)/fs;
min_err = Inf;
min_na = NaN;
for n = na
    model = models{n};
    A = model{1}; B = model{2}; C = model{3}; D = model{4};
    [A,B,C] = dbalreal(A,B,C);  % Balance realizations
    yval_hat = lsim(ss(A,B,C,D,1/fs), uval, tval);

    err = yval - yval_hat; 
    err = sqrt(mean(err(end-Nt+1:end).^2));
    if err < min_err
        min_na = n;
        min_err = err;
    end
end
% Select the best model
model = models{min_na};
[A,B,C,D] = model{:};
[A,B,C] = dbalreal(A,B,C);

%% Misc Settings
% Transients
Ntrans = Nt;
T1 = [Ntrans 1+(0:Nt:(R-1)*Nt)];
T2 = 0;

m = size(u,2);
p = size(y,2);

nx = [3];
ny = [];
whichtermsx = 'statesonly';
whichtermsy = 'empty';

% Optimization
MaxCount = 100;
lambda = 100;

% model order 
n = min_na;

% BLA models with zero nonlinear coefficients
model = fCreateNLSSmodel(A, B, C, D, nx, ny, T1, T2);
model.xactive = fSelectActive(whichtermsx, n, m, n, nx);
model.yactive = fSelectActive(whichtermsy, n, m, p, ny);

modellinest = model;
modellinval = model;  modellinval.T1 = Nt;
modellintest = model; modellintest.T1 = 0;

% Weighting
W = [];  

%% Sequential PNLSS
% errormeasures = cell(size(Alevels));
% seqmodels = cell(size(Alevels));
% modelguess = modellinest;
for ia=2:length(Alevels)
    load(sprintf('./data/ode45_multisine_A%.2f_F%d_mds%d.mat',Alevels(ia), fs, mds), 'u', 'y');

    [Nt, P, R, n] = size(y);
	freq = (0:Nt-1)*f0;
    % Use middle deflection
%     y = PHI_L2*reshape(y, [], n)';
    yout = sum(y.*permute(PHI_L2, [1 3 4 2]),4);
    y = yout;
    y = reshape(y, [Nt, P, R]);
    
    % Separate data
    utest = u(:, end, end);  utest = utest(:);
    ytest = y(:, end, end);  ytest = ytest(:);                

    uval = u(:, end, end-1);  uval = uval(:);
    yval = y(:, end, end-1);  yval = yval(:);
        
    if PerOrNot(ia)   % if periodic
        % kick off transient
        Ptr = 6;
        P = P-Ptr;
        R = R-2;
        u = u(:, Ptr:end, 1:R);
        y = y(:, Ptr:end, 1:R);

        u = permute(u, [1,4,3,2]);  % N x m x R x P
        y = permute(y, [1,4,3,2]);  % N x p x R x P
        covY = fCovarY(y);  % Noise covariance (Frequency domain)

        u = mean(u, 4);
        y = mean(y, 4);
        
        modelguess.T1 = [Ntrans 1+(0:Nt:(R-1)*Nt)];
    else  % if not periodic
        % No concept of transients
        Ptr = P-1;
        P = P-Ptr;
        R = R-2;
        u = u(:, Ptr+1:end, 1:R);
        y = y(:, Ptr+1:end, 1:R);

        % Stack all the periods together since 
%         uc = u(:);
%         yc = y(:);
        u = reshape(u, [Nt*P, 1, R]);
        y = reshape(y, [Nt*P, 1, R]);
        
        u = permute(u, [1,4,3,2]);  % N x m x R x P
        y = permute(y, [1,4,3,2]);  % N x p x R x P
        covY = fCovarY(y);  % Noise covariance (Frequency domain)
        
%         % Non-periodic input
%         modelguess.T1 = [0];
%         modelguess.T2 = Nt*(Ptr-1);

        Nt = size(u,1);
        
        % Periodic input
        modelguess.T1 = [Ptr*Ntrans 1+(0:Nt:(R-1)*Nt)];
%         modelguess.T1 = [0];
        modelguess.T2 = [];
    end
    
    m = size(u, 2);
    p = size(y, 2);
    uc = u(:);
    yc = y(:);
    
    % Simulate others
    y_linest = fFilterNLSS(modellinest, uc);
    err_linest = yc-y_linest;
    
    y_linval = fFilterNLSS(modellinval, uc);
    err_linval = yc-y_linval;
    
    y_lintest = fFilterNLSS(modellintest, uc);
    err_lintest = yc-y_lintest;
    
    y_nlinit = fFilterNLSS(model, uc);
    err_nlinit = yc-y_nlinit;

    % PNLSS optimization
    try
        if PerOrNot(ia)
            [~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, modelguess, MaxCount, W, lambda);
        else
            modelguess.x0active = (1:modelguess.n)';
            modelguess.u0active = (1:modelguess.m)';
            [~, y_mod, models_pnlss] = fLMnlssWeighted_x0u0(uc, yc, modelguess, MaxCount, W, lambda);
%         [~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, modelguess, MaxCount, W, lambda);
        end
    catch 
        modelguess.E = modelguess.E*0;
        [~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, modelguess, MaxCount, W, lambda);
    end
%  	[~, y_mod, models_pnlss] = fLMnlssWeighted(uc, yc, models_pnlss(end-5), MaxCount, W, lambda);
    err_nlest = yc-y_mod;
    
    % Choose best model from PNLSS (using validation data)
    valerrs = [];
    for i=1:length(models_pnlss)
        if PerOrNot(ia)
            models_pnlss(i).T1 = Ntrans;
        else
            models_pnlss(i).T1 = 0;
            models_pnlss(i).T2 = Nt*(Ptr-1);
        end
        yval_mod = fFilterNLSS(models_pnlss(i), uval);
        valerr = yval - yval_mod;
        valerrs = [valerrs; rms(valerr)];
    end
    
    [min_valerr, i] = min(valerrs);
    model = models_pnlss(i);    % Updated PNLSS model
    modelguess.A = model.A;
    modelguess.B = model.B;
    modelguess.C = model.C;
    modelguess.D = model.D;
    modelguess.E = model.E;
    modelguess.F = model.F;

    % Compute output error on validation data
    modelval = model;  modelval.T1 = Ntrans;
    y_nlval = fFilterNLSS(modelval, uval);
    err_nlval = yval-y_nlval;
    % Compute output error on test data
    modeltest = model;  modeltest.T1 = Ntrans;
    y_nltest = fFilterNLSS(modeltest, utest);
    err_nltest = ytest-y_nltest;
    
    err = ...
        [rms(err_linest), rms(err_nlest); ...
        rms(err_linval), rms(err_nlval);...
        rms(err_lintest), rms(err_nltest)];
    
    errormeasures{ia} = err;
    seqmodels{ia} = model;
    
    save(sprintf('./data/pnlssmodel_A%.2f_F%d_nx%s_mds%d.mat', Alevels(ia), fs, sprintf('%d',nx),mds), 'model', 'err')
    
    fprintf('Done %d/%d\n', ia, length(Alevels))
    
    % Plot

    % Estimation data
    figure(ia);
    clf()
	plottime = [yc err_linest err_nlest];
    t = ((1:size(plottime,1))-1)*t(2);
    plot(t(1:Nt*R),plottime)
    xlabel('Time (s)')
    ylabel('Output (errors)')
    legend('output','linear error','PNLSS error')
    title('Estimation results')
    print(sprintf('./fig/TDOMESTRESS_PNLSS_A%.2f_F%d_nx%s_mds%d.eps', Alevels(ia), fs, sprintf('%d',nx), mds), '-depsc')
    disp(' ')
    disp(['rms(y-y_mod) = ' num2str(rms(yc-y_mod))...
        ' (= Output error of the best PNLSS model on the estimation data)'])
    % disp(['rms(noise(:))/sqrt(P) = ' num2str(rms(noise(:))/sqrt(P)) ' (= Noise level)'])
end

save(sprintf('./data/seqpnlssmodels_mds%d.mat',mds), 'Alevels', 'errormeasures', 'seqmodels');