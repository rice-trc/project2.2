clc
clear all
addpath('../src/mhbm_res/')
addpath('../src/nlvib/SRC/')

load('./data/ode45_multisine.mat', 'fs', 't', 'u')
load('./data/pnlssout_try0.mat','model');

%%
Nd = size(model.A,1);
Nh = 3;
Ntd = 2^10;
Ur = zeros(2*Nh+1, 1);
Ur(2) = 1;

F = 3;

Uc = Ur([1 2:2:end]) - 1j*[0; Ur(3:2:end)];

Ur = Ur*F;
Uc = Uc*F;

%% Run with NLvib
Ws = 200*2*pi;
We = 360*2*pi;
ds = 1*2*pi;
dsmin = 0.001*2*pi;
dsmax = 50*2*pi;
% X0 = rand((2*Nh+1)*size(model.A,1),1);

Xc = (exp(1i*Ws/fs)*eye(size(model.A))-model.A)\(model.B*F);             % linear solution
% Xc = ((1i*Ws/fs)*eye(size(model.A))-model.A)\(model.B*F);             % linear solution
X0 = [zeros(length(model.A),1);real(Xc);-imag(Xc);....
        zeros(2*(Nh-1)*length(model.A),1)];                  % initial guess
    
Dscale = [mean(abs(Xc))*ones(length(X0),1);Ws];
Sopt = struct('ds',ds,'dsmin',dsmin,'dsmax',dsmax,'flag',1,'stepadapt',1, ...
        'predictor','tangent','parametrization','arc_length', ...
        'Dscale',Dscale,'jac','full', 'dynamicDscale', 1);

fun_residual = ...
        @(XX) mhbm_aft_residual_pnlss_discrete(XX, model.A, model.B, model.E, model.xpowers, 1/fs, Uc, Nh, Ntd);
Cfun_postprocess = {@(varargin) ...
        mhbm_post_amplitude_pnlss(varargin{:},Uc,model.C,model.D,zeros(1,length(model.E)),model.xpowers,Nh,Ntd)};
fun_postprocess = @(Y) mhbm_postprocess(Y,fun_residual,Cfun_postprocess);

[Xr,~,Sol] = solve_and_continue(X0, fun_residual,...
    Ws, We, ds, Sopt, fun_postprocess);
A_hb = [Sol.Apv];
Ph_hb = [Sol.Aph1];
W_hb = Xr(end,:);

figure(1)
subplot(2,1,1);
hold on;
plot(W_hb/2/pi, A_hb, '.-')
subplot(2,1,2);
hold on;
plot(W_hb/2/pi, Ph_hb, '.-');
ylim([-180 0])