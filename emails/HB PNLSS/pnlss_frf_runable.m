clc
clear
addpath('NLvib_v1.1/SRC')
addpath('NLvib_v1.1/SRC/mhbm_res/')

load('ode45_multisine.mat', 'fs')
load('pnlssout_try0.mat','model');

%%
Nh = 3;
Ntd = 2^10;
Ur = zeros(2*Nh+1, 1);

F = 50;

Ur(2) = F;
Uc = Ur([1 2:2:end]) - 1j*[0; Ur(3:2:end)];

%% Run with NLvib
Ws = 240*2*pi;
We = 320*2*pi;
ds = 1;


Xc = (exp(1i*Ws/fs)*eye(size(model.A))-model.A)\(model.B*F);             % linear solution
X0 = [zeros(length(model.A),1);real(Xc);-imag(Xc);....
        zeros(2*(Nh-1)*length(model.A),1)];                  % initial guess
% X0 = rand((2*Nh+1)*size(model.A,1),1);
Sopt = struct('ds',ds,'dsmin',ds/5,'dsmax',5*ds,'flag',1,'stepadapt',1, ...
        'predictor','tangent','parametrization','arc_length', ...
        'Dscale',[10*ones(length(X0),1);Ws],'jac','full');

fun_residual = ...
        @(XX) mhbm_aft_residual_pnlss_discrete(XX, model.A, model.B, model.E, model.xpowers, 1/fs, Uc, Nh, Ntd);
Cfun_postprocess = {@(varargin) ...
        mhbm_post_amplitude_pnlss(varargin{:},Uc,model.C,model.D,zeros(1,length(model.E)),model.xpowers,Nh,Ntd)};
fun_postprocess = @(Y) mhbm_postprocess(Y,fun_residual,...
        Cfun_postprocess);
    
[Xr,~,Sol] = solve_and_continue(X0, fun_residual,...
    Ws, We, ds, Sopt,fun_postprocess);
% Xc = Xr([1 2:2:end],:) - 1j*[Xr(1,:)*0; Xr(3:2:end, :)]; % that's wrong!
% test1 = Xr(3,:) - 1j*Xr(4, :);
A_hb  = [Sol.Amax];
% Sol = [];
% model.F = model.E(1,:)*0;
W_hb = Xr(end,:);
% As = zeros(size(Xr(end,:)));
% for i=1:size(Xr,2)
%     Sol = mhbm_post_amplitude_pnlss(Sol, Xc(:,i), 1, Uc, model.C, model.D, model.F, model.xpowers, Nh, Ntd);
%     As(i) = Sol.A;
% end

% Yr = kron(eye(2*Nh+1), model.C)*Xr(1:end-1,:);
% YAmp = sqrt([1 0.5*ones(1,2*Nh)]*Yr.^2)

figure
plot(W_hb/2/pi,A_hb, '.-')
% hold on
% plot(Ws/2/pi,abs(test1),'.--')
