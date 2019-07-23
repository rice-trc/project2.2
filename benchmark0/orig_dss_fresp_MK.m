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

% set nonlinear coefficient to 0
fname = ['beam_New_Design_Steel_analytical_5t_' ...
    num2str(thickness*1000) 'mm.mat'];
p = 3;
E = 0;
% E = 5.7085e14;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(1:Nmod).*om(1:Nmod));
K = diag(om.^2);

% M = 1.1;
% K = 1.25;
% D = 0.01;
% om = sqrt(K/M);

% Fundamental harmonic of external forcing
Fex1 = gam;
% Fex1 = 1;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);

% Number of degrees of freedom
n = oscillator.n;

%% Frequency response of original system using HBM
analysis = 'FRF';
H = 1;              % harmonic order
N=2*3*H+1;

% Analysis parameters
Om_s = .75*om;      % start frequency
Om_e = 1.25*om;     % end frequency

% Excitation levels
% exc_lev = 1;
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
    
%     % Solve and continue w.r.t. Om
%     ds = 50; % -> better for exc_lev = 50
        
    TYPICAL_x = oscillator.Fex1/(2*D*M*om^2);
    Dscale = [1e0*TYPICAL_x*ones(length(y0),1);(Om_s+Om_e)/2];
    % arc-length continuation
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1); ds = 1e2;
%     % sequential continuation
%     Sopt = struct('flag',0,'Dscale',Dscale,'dynamicDscale',1,'stepadapt',0); ds = abs(Om_s-Om_e)/200;
    
    X{iex} = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,H,N,analysis),...
        Om_s,Om_e,ds,Sopt);
	Sols{iex} = [X{iex}(end,:)' sqrt([1 0.5*ones(1,2*H)]*X{iex}(1:end-1,:).^2)'...
        atan2d(-X{iex}(3,:),X{iex}(2,:))'];
end

%% Compute frequency response of PNLSS identified model
% load('./data/ode45_multisine_A25.mat', 'fs');
% load('./data/pnlssout_A25.mat','model');
% ctmodel.xpowers = model.xpowers;
% ctmodel.E = [[0; -M\E] zeros(2, 9)];

Ntd = 2^6;
% samples_per_pseudoperiod = 1e4; Om_ref = (Om_e+Om_s)/2;
% fs = Om_ref/(2*pi)*samples_per_pseudoperiod;

% sampling of 4096 - DOESN'T WORK
fs = 2^12;

% Continuous time model
ctmodel.A = [0 1;
            -M\K -M\D];
ctmodel.B = [0; M\Fex1];
ctmodel.C = [1 0];
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
%% Validate in time domain
% Om = .98*om;
% y0 = rand(2,1);
% Ns = 100;
% tend = 2*pi/Om*500;
% sol_ct = ode45(@(t,y) ctmodel.A*y + ctmodel.B*cos(Om*t),[0 tend],y0,...
%     odeset('MaxStep',2*pi/Om/Ns));
% 
% % analytical discretization for A,B
% ts = 2*pi/Om/Ns;
% fs = 1/ts;
% dtmodel.A = expm(ctmodel.A/fs);
% dtmodel.B = ctmodel.A\(dtmodel.A-eye(2))*ctmodel.B;
% dtmodel.E = ctmodel.E/fs;
% 
% N_dt = round(tend/ts);
% Y_dt = zeros(size(sol_ct.y,1),N_dt+1);
% t_dt = 0:ts:N_dt*ts;
% Y_dt(:,1) = y0;
% for k=1:N_dt
%     Y_dt(:,k+1) = dtmodel.A*Y_dt(:,k) + dtmodel.B*cos(Om*t_dt(k));
% end
% 
% figure; hold on;
% plot(sol_ct.x,sol_ct.y(2,:),'k-');
% plot(t_dt,Y_dt(2,:),'g--');
%%
Ndpnlss = size(dtmodel.A,1);
% Forcing vector
Uc = zeros(H+1,1);
Uc(2) = 1;

% ds = 1*2*pi;
% dsmin = 0.001*2*pi;
% dsmax = 50*2*pi;

Wstart = Om_s;
Wend   = Om_e;

Xpnlss = cell(length(exc_lev),1);
Solspnlss = cell(length(exc_lev),1);
for iex=1:length(exc_lev)
    Ff = exc_lev(iex);

    Xc = (exp(1i*Wstart/fs)*eye(size(dtmodel.A))-dtmodel.A)\(dtmodel.B*Ff);             % linear solution
    % Xc = ((1i*Wstart/fs)*eye(size(dtmodel.A))-dtmodel.A)\(dtmodel.B*Ff);             % linear solution
    X0 = [zeros(length(dtmodel.A),1);real(Xc);-imag(Xc);....
            zeros(2*(H-1)*length(dtmodel.A),1)];                  % initial guess
    
    TYPICAL_x = 1e0*Ff/(2*D*M*om^2);
    Dscale = [1*TYPICAL_x*ones(length(X0),1);(Wstart+Wend)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1); ds = 1e2;
%     Sopt = struct('flag',0,'stepadapt',0,'Dscale',Dscale,'jac','full','dynamicDscale', 1); ds = abs(Om_s-Om_e)/200;
%     Sopt = struct('ds',ds,'dsmin',dsmin,'dsmax',dsmax,'flag',1,'stepadapt',1, ...
%             'predictor','tangent','parametrization','arc_length', ...
%             'Dscale',Dscale,'jac','none', 'dynamicDscale', 1);

    fun_residual = ...
            @(XX) mhbm_aft_residual_pnlss_discrete(XX, dtmodel.A, ...
            dtmodel.B, dtmodel.E, dtmodel.xpowers, 1/fs, Uc*Ff, H, Ntd);
    Cfun_postprocess = {@(varargin) ...
            mhbm_post_amplitude_pnlss(varargin{:},Uc*Ff,dtmodel.C,...
            dtmodel.D,dtmodel.F,dtmodel.ypowers,H,Ntd)};
    fun_postprocess = @(Y) mhbm_postprocess(Y,fun_residual,Cfun_postprocess);

    [Xpnlss{iex},~,Sol] = solve_and_continue(X0, fun_residual,...
        Wstart, Wend, ds, Sopt, fun_postprocess);
    Solspnlss{iex} = [Xpnlss{iex}(end,:)' [Sol.Apv]' [Sol.Aph1]'];
end

% Plot
fg1 = 3000;
fg2 = 4000;

figure(fg1)
clf()

figure(fg2)
clf()
colos = distinguishable_colors(length(exc_lev));
aa = gobjects(size(exc_lev));
for iex=1:length(exc_lev)
    figure(fg1)
    plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-x', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2), '.--', 'Color', colos(iex,:))
    
    figure(fg2)
    aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3), '-x', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3), '.--', 'Color', colos(iex,:))
    legend(aa(iex), sprintf('F = %.2f', exc_lev(iex)));
end

figure(fg1)
set(gca,'Yscale', 'log')
xlim(sort([Om_s Om_e])/2/pi)
xlabel('Forcing frequency \omega (Hz)')
ylabel('RMS response amplitude (m)')
% savefig(sprintf('./fig/pnlssfrf_A%d_Amp.fig',Alevel))
% print(sprintf('./fig/pnlssfrf_A%d_Amp.eps',Alevel), '-depsc')
print('./fig/send_amp.png', '-dpng')

figure(fg2)
xlim(sort([Om_s Om_e])/2/pi)
xlabel('Forcing frequency \omega (Hz)')
ylabel('Response phase (degs)')
legend(aa(1:end), 'Location', 'northeast')
% savefig(sprintf('./fig/pnlssfrf_A%d_Phase.fig',Alevel))
% print(sprintf('./fig/pnlssfrf_A%d_Phase.eps',Alevel), '-depsc')
print('./fig/send_phase.png', '-dpng')