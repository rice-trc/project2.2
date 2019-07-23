clc
clear all
addpath('../src/mhbm_res/')
addpath('../src/matlab/')
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')
addpath('../src/nlvib_hillsexp/')

%% Define system

% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.01;
Nmod = 1;
setup = './data/New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
fname = ['./data/beam_New_Design_Steel_analytical_5t_' ...
    num2str(thickness*1000) 'mm.mat'];
[p, E] = nlcoeff(fname, Nmod);

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(1:Nmod).*om(1:Nmod));
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = PHI_L2;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);

% Number of degrees of freedom
n = oscillator.n;

%% Frequency response of original system using HBM
analysis = 'FRF';
H = 7;              % harmonic order
N=2*3*H+1;
Ntd = 1e3;

% Analysis parameters
Om_s = 200*2*pi;      % start frequency
Om_e = 700*2*pi;     % end frequency

% Excitation levels
% exc_lev = [10 40 60 80 100];
exc_lev = [0.1 2.5 7.5 10 15];
exc_lev = [1e-2 3e-2 5e-2 8e-2 1e-1];
X = cell(size(exc_lev));
Sols = cell(size(exc_lev));
HillsExps = cell(size(exc_lev));
for iex=1:length(exc_lev)
    % Set excitation level
    oscillator.Fex1 = Fex1*exc_lev(iex);
    
    % Initial guess (solution of underlying linear system)
    Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
    y0 = zeros((2*H+1)*length(Q1),1);
    y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
    qscl = max(abs((-om(1)^2*M + 1i*om(1)*D + K)\oscillator.Fex1));
    
    % Solve and continue w.r.t. Om
    ds = 100;
        
    TYPICAL_x = oscillator.Fex1/(2*D*M*om^2);
    Dscale = [TYPICAL_x*ones(length(y0),1);(Om_s+Om_e)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4);
    [X{iex},~,HillsExps{iex}] = solve_and_continue(y0, ...
        @(X) HB_residual(X,oscillator,H,N,analysis),...
        Om_s,Om_e,ds,Sopt, @(Y) HB_hillsexp_nlvib(Y, @(X) HB_residual(X,oscillator,H,N,analysis), oscillator.M, oscillator.D, H, Nmod));
	Sols{iex} = [X{iex}(end,:)' PHI_L2*sqrt([1 0.5*ones(1,2*H)]*X{iex}(1:end-1,:).^2)'...
        atan2d(-X{iex}(3,:),X{iex}(2,:))' [HillsExps{iex}.stab]' [HillsExps{iex}.unstab]'];
end

%% Compute frequency response of PNLSS identified model
% N = 1e3;
exc_lev = [1e-2 3e-2 5e-2 8e-2 1e-1];
% Alevels = 0.5*N*exc_lev.^2;
Alevels = [0.01 0.05 0.10 0.15 0.20 0.25];
nx = [3];

for ia = 4 % 1:length(Alevels)
Alevel = Alevels(ia);
fs = 4096;
load(sprintf('./data/ode45_multisine_A%.2f_F%d.mat',Alevel,fs), 'PHI_L2');
load(sprintf('./data/pnlssmodel_A%.2f_F%d_nx%s.mat',Alevel,fs,sprintf('%d',nx)),'model');
% load(sprintf('./data/pnlssout_A%d_F%d.mat',Alevel,fs),'model');

Ndpnlss = size(model.A,1);

% Forcing vector
Uc = zeros(H+1,1);
Uc(2) = 1;

ds = 1*2*pi;
dsmin = 0.0001*2*pi;
dsmax = 100*2*pi;

Xpnlss = cell(length(exc_lev),1);
Solspnlss = cell(length(exc_lev),1);
for iex=1:length(exc_lev)
    Ff = exc_lev(iex);

    Xc = (exp(1i*Om_s/fs)*eye(size(model.A))-model.A)\(model.B*Ff);             % linear solution
    % Xc = ((1i*Om_s/fs)*eye(size(model.A))-model.A)\(model.B*Ff);             % linear solution
    X0 = [zeros(length(model.A),1);real(Xc);-imag(Xc);....
            zeros(2*(H-1)*length(model.A),1)];                  % initial guess
    
% 	TYPICAL_x = 1e5*Ff/(2*D*M*om^2);
    TYPICAL_x = 1e-4;
    Dscale = [TYPICAL_x*ones(length(X0),1);Om_s];
    Sopt = struct('ds',ds,'dsmin',dsmin,'dsmax',dsmax,'flag',1,'stepadapt',1, ...
            'predictor','tangent','parametrization','arc_length', ...
            'Dscale',Dscale,'jac','full', 'dynamicDscale', 1, 'stepmax', 2000);

    fun_residual = ...
            @(XX) mhbm_aft_residual_pnlss_discrete(XX, model.A, model.B, model.E, model.xpowers, 1/fs, Uc*Ff, H, Ntd);
    Cfun_postprocess = {@(varargin) ...
            mhbm_post_amplitude_pnlss(varargin{:},Uc*Ff,model.C,model.D,zeros(1,length(model.E)),model.xpowers,H,Ntd)};
    fun_postprocess = @(Y) mhbm_postprocess(Y,fun_residual,Cfun_postprocess,H,model.n,fs);

    [Xpnlss{iex},~,Sol] = solve_and_continue(X0, fun_residual,...
        Om_s, Om_e, ds, Sopt, fun_postprocess);
    Solspnlss{iex} = [Xpnlss{iex}(end,:)' [Sol.Apv]' [Sol.Aph1]' [Sol.stab]' [Sol.unstab]'];
end

%% Plot
figure(10*ia)
clf()

figure(10*ia+1)
clf()
colos = distinguishable_colors(length(exc_lev));
aa = gobjects(size(exc_lev));
for iex=1:length(exc_lev)
    figure(10*ia)
%     plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2).*Sols{iex}(:,4), '-', 'Color', colos(iex,:)); hold on
%     plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2).*Sols{iex}(:,5), '--', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2).*Solspnlss{iex}(:,4), '.-', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2).*Solspnlss{iex}(:,5), '+-', 'Color', colos(iex,:)); hold on
    
    figure(10*ia+1)
%     aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3).*Sols{iex}(:,4), '-', 'Color', colos(iex,:)); hold on
%     plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3).*Sols{iex}(:,5), '--', 'Color', colos(iex,:)); hold on
    aa(iex) = plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3).*Solspnlss{iex}(:,4), '.-', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3).*Solspnlss{iex}(:,5), '+-', 'Color', colos(iex,:)); hold on
    legend(aa(iex), sprintf('F = %.2f', exc_lev(iex)));
end

figure(10*ia)
set(gca, 'YScale', 'log')
xlim(sort([Om_s Om_e])/2/pi)
xlim([200 350])
xlabel('Forcing frequency \omega (Hz)')
ylabel('RMS response amplitude (m)')
% savefig(sprintf('./fig/pnlssfrf_A%d_Amp.fig',Alevels(ia)))
% print(sprintf('./fig/pnlssfrf_A%.2f_Amp_nx%s.eps',Alevels(ia),sprintf('%d',nx)), '-depsc')
% print('./fig/stabsol_Amp.eps', '-depsc')
print('./fig/dtstabsol_Amp.eps', '-depsc')

figure(10*ia+1)
xlim(sort([Om_s Om_e])/2/pi)
xlim([200 350])
xlabel('Forcing frequency \omega (Hz)')
ylabel('Response phase (degs)')
legend(aa(1:end), 'Location', 'northeast')
% savefig(sprintf('./fig/pnlssfrf_A%d_Phase.fig',Alevels(ia)))
% print(sprintf('./fig/pnlssfrf_A%.2f_Phase_nx%s.eps',Alevels(ia),sprintf('%d',nx)), '-depsc')
% print('./fig/stabsol_Phase.eps', '-depsc')
print('./fig/dtstabsol_Phase.eps', '-depsc')
end