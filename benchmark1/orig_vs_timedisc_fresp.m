clc
clear all
addpath('../src/mhbm_res/')
addpath('../src/matlab/')
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')

%% Define system

% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.01;
Nmod = 1;
setup = 'New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
fname = ['beam_New_Design_Steel_analytical_5t_' ...
    num2str(thickness*1000) 'mm.mat'];
[p, E] = nlcoeff(fname, Nmod);
% E = 0;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(1:Nmod).*om(1:Nmod));
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = gam;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);

% Number of degrees of freedom
n = oscillator.n;

%% Frequency response of original system using HBM
analysis = 'FRF';
H = 1;              % harmonic order
N=2*3*H+1;
Ntd = 2^6;

% Analysis parameters
Om_s = 200*2*pi;      % start frequency
Om_e = 750*2*pi;     % end frequency

% Excitation levels
exc_lev = [10 40 60 80 100];
X = cell(size(exc_lev));
Sols = cell(size(exc_lev));
for iex=1:length(exc_lev)
    % Set excitation level
    oscillator.Fex1 = Fex1*exc_lev(iex);
    
    % Initial guess (solution of underlying linear system)
    Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
    y0 = zeros((2*H+1)*length(Q1),1);
    y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
    qscl = max(abs((-om(1)^2*M + 1i*om(1)*D + K)\oscillator.Fex1));
    
    % Solve and continue w.r.t. Om
    ds = 100; % -> better for exc_lev = 50
    
    TYPICAL_x = oscillator.Fex1/(2*D*M*om^2);
    Dscale = [TYPICAL_x*ones(length(y0),1);(Om_s+Om_e)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4);
    X{iex} = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,H,N,analysis),...
        Om_s,Om_e,ds,Sopt);
	Sols{iex} = [X{iex}(end,:)' PHI_L2*sqrt([1 0.5*ones(1,2*H)]*X{iex}(1:end-1,:).^2)'...
        atan2d(-X{iex}(3,:),X{iex}(2,:))'];
end

%% Compute frequency response of PNLSS identified model
% load('./data/ode45_multisine_A25.mat', 'fs');
% load('./data/pnlssout_A25.mat','model');
% ctmodel.xpowers = model.xpowers;
% ctmodel.E = [[0; -M\E] zeros(2, 9)];

Ntd = 2^6;
fs = 2^28;

% Continuous time model
ctmodel.A = [0 1;
            -M\K -M\D];
ctmodel.B = [0; M\Fex1];
ctmodel.C = [PHI_L2 0];
ctmodel.D = [0];
% xpowers: each row gives the power of [x1, x2, u], which is then
% multiplied with corresponding coefficient in E.
% E ∈ [n, nx], xpowers ∈ [nx, n+1]
ctmodel.xpowers = [3 0 0];
ctmodel.E = [0; -M\E];
ctmodel.F = 0;
ctmodel.ypowers = [3 0 0];


% Discrete time model
% euler discretization
dtmodel = ctmodel;
dtmodel.A = ctmodel.A/fs+eye(2);
dtmodel.B = ctmodel.B/fs;
dtmodel.E = ctmodel.E/fs;

% analytical discretization for A,B
dtmodel.A = expm(ctmodel.A/fs);
dtmodel.B = ctmodel.A\(dtmodel.A-eye(2))*ctmodel.B;
dtmodel.E = ctmodel.E/fs;

% matlabs build-in / for checking
dtm = c2d(ss(ctmodel.A,ctmodel.B,ctmodel.C,ctmodel.D),1/fs);

Ndpnlss = size(dtmodel.A,1);
% Forcing vector
Uc = zeros(H+1,1);
Uc(2) = 1;

ds = 1*2*pi;
dsmin = 0.001*2*pi;
dsmax = 200*2*pi;

Wstart = Om_e;
Wend   = Om_s;

Xpnlss = cell(length(exc_lev),1);
Solspnlss = cell(length(exc_lev),1);
for iex=1:length(exc_lev)
    Ff = exc_lev(iex);

    Xc = (exp(1i*Wstart/fs)*eye(size(dtmodel.A))-dtmodel.A)\(dtmodel.B*Ff);             % linear solution
    % Xc = ((1i*Wstart/fs)*eye(size(dtmodel.A))-dtmodel.A)\(dtmodel.B*Ff);             % linear solution
    X0 = [zeros(length(dtmodel.A),1);real(Xc);-imag(Xc);....
            zeros(2*(H-1)*length(dtmodel.A),1)];                  % initial guess
    
	TYPICAL_x = 1e0*Ff/(2*D*M*om^2);
%     TYPICAL_x = 1e-2;

% ds = 1*2*pi;
% dsmin = 0.001*2*pi;
% dsmax = 10*2*pi;

    Dscale = [TYPICAL_x*ones(length(X0),1);Wstart];
    Sopt = struct('ds',ds,'dsmin',dsmin,'dsmax',dsmax,'flag',1,'stepadapt',1, ...
            'predictor','tangent','parametrization','arc_length', ...
            'Dscale',Dscale,'jac','full', 'dynamicDscale', 1);

    fun_residual = ...
            @(XX) mhbm_aft_residual_pnlss_discrete(XX, dtmodel.A, ...
            dtmodel.B, dtmodel.E, dtmodel.xpowers, 1/fs, ...
            Uc*Ff, H, Ntd);
    Cfun_postprocess = {@(varargin) ...
            mhbm_post_amplitude_pnlss(varargin{:},Uc*Ff,dtmodel.C,dtmodel.D,dtmodel.F,dtmodel.ypowers,H,Ntd)};
    fun_postprocess = @(Y) mhbm_postprocess(Y,fun_residual,Cfun_postprocess);

    [Xpnlss{iex},~,Sol] = solve_and_continue(X0, fun_residual,...
        Wstart, Wend, ds, Sopt, fun_postprocess);
    Solspnlss{iex} = [Xpnlss{iex}(end,:)' [Sol.Apv]' [Sol.Aph1]'];
end

% %% Plot
fg1 = 100;
fg2 = 200;

figure(fg1)
clf()

figure(fg2)
clf()
colos = distinguishable_colors(length(exc_lev));
aa = gobjects(size(exc_lev));
for iex=1:length(exc_lev)
    figure(fg1)
    plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2), '.--', 'Color', colos(iex,:))
    
    figure(fg2)
    aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3), '-', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3), '.--', 'Color', colos(iex,:))
    legend(aa(iex), sprintf('F = %.2f', exc_lev(iex)));
end

figure(fg1)
set(gca,'Yscale', 'linear')
xlim(sort([Om_s Om_e])/2/pi)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('RMS response amplitude (m)')
% savefig(sprintf('./fig/pnlssfrf_A%d_Amp.fig',Alevel))
print(sprintf('./fig/dssex_frf_Amp_fs%d.eps',fs), '-depsc')

figure(fg2)
xlim(sort([Om_s Om_e])/2/pi)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
legend(aa(1:end), 'Location', 'northeast')
% legend(aa(1:end), 'Location', 'best')
% savefig(sprintf('./fig/pnlssfrf_A%d_Phase.fig',Alevel))
print(sprintf('./fig/dssex_frf_Phase_fs%d.eps',fs), '-depsc')