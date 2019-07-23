clc
clear all
addpath('../src/mhbm_res/')
addpath('../src/matlab/')
addpath('../src/nlvib/SRC/')
addpath('../src/pnlss/')
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
E = 5e14;

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
H = 7;              % harmonic order
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
    ds = 1e2; % -> better for exc_lev = 50
        
    TYPICAL_x = oscillator.Fex1/(2*D*M*om^2);
    Dscale = [TYPICAL_x*ones(length(y0),1);(Om_s+Om_e)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4);
    X{iex} = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,H,N,analysis),...
        Om_s,Om_e,ds,Sopt);
	Sols{iex} = [X{iex}(end,:)' PHI_L2*sqrt([1 0.5*ones(1,2*H)]*X{iex}(1:end-1,:).^2)'...
        atan2d(-X{iex}(3,:),X{iex}(2,:))'];
end

%% Compute frequency response of Discretized model
% load('./data/ode45_multisine_A25.mat', 'fs');
% load('./data/pnlssout_A25.mat','model');
% ctmodel.xpowers = model.xpowers;
% ctmodel.E = [[0; -M\E] zeros(2, 9)];

Ntd = 2^6;
% samples_per_pseudoperiod = 1e4; Om_ref = (Om_e+Om_s)/2;
% fs = Om_ref/(2*pi)*samples_per_pseudoperiod;
fs = 2^12;

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

dm = c2d(ss(ctmodel.A, ctmodel.B, ctmodel.C, ctmodel.D), 1/fs, 'tustin');

dtmodel.A = dm.A;
dtmodel.B = dm.B;
dtmodel.C = dm.C;
dtmodel.D = dm.D;

% parameters for PNLSS. Initialize as linear. We change this later
% nx/ny is not needed for time simulation
nx = []; ny = []; T1 = 0; T2= 0;
model = fCreateNLSSmodel(dtmodel.A,dtmodel.B,dtmodel.C,dtmodel.D,nx,ny,T1,T2);
model.E = dtmodel.E;
model.F = dtmodel.F;
model.xpowers = dtmodel.xpowers;
model.ypowers = dtmodel.ypowers;

%% simulate model -- From here is new!
% direction to sweep
dir = 1; % or -1

% We see each increment in frequency as a new realization.
% Vector of excited frequencies. In Hz!
Om_vec = linspace(200,350,25);
Om_vec = [linspace(200, 260, 10) linspace(260, 270, 20) linspace(270, 350, 10)];

if dir == -1
    Om_vec = fliplr(Om_vec);
end
R = length(Om_vec);
P = 20;

% periods to calculate max amp over.
nper = 5;

%% Simulate with constant fs, non-integer periods
% this is the only way to simulate the nolinear sys. Slow!

Nt = 1e3;
Ptrans = 150;
nsim = fs/Nt;
T = linspace(0, (P+Ptrans)/nsim, Nt*(P+Ptrans)+1)'; T(end) = [];

a_max = zeros(R,length(exc_lev));

a_rms = zeros(R,length(exc_lev));
ph_rs = zeros(R,length(exc_lev));

a_rms_R = zeros(R,length(exc_lev));
ph_rs_R = zeros(R,length(exc_lev));

j = 1;
Ntint = 2^4;
Yint = zeros(Nt,1);
y = zeros(Ntint*P,R);
for A = exc_lev
    X0 = zeros(length(model.A),1);
    for i = 1:R
        Om = Om_vec(i);
        u = A*sin(2*pi*Om*T);
%         [Y, ~, X] = lsim(ss(model.A,model.B,model.C,model.D,-1),u,[],X0);
        [Y, X] = fFilterNLSS(model,u,X0); % Modeled output signal
        % [Y, ~, X] = lsim(ss(ctmodel.A,ctmodel.B,ctmodel.C,ctmodel.D),u,T,X0);
        X0 = X(end,:);
        ymax = max(Y(end-nper*Nt+1:end));
        ymin = min(Y(end-nper*Nt+1:end));
        a_max(i,j) = abs(0.5*(ymax-ymin));
        
        Pint = floor(T(end)*Om);
        Tint = linspace(0, Pint/Om, Ntint*Pint+1)';  Tint(end) = [];
        Urint = A*sin(2*pi*Om*Tint(1:Ntint));
        Ufint = fft(Urint)/(Ntint/2); Ufint(1) = Ufint(1)/2;
        
        % REGRESSION APPROACH
        YR = [ones(size(T(end-(P*Nt)+1:end))) cos(2*pi*Om*T(end-(P*Nt)+1:end)) sin(2*pi*Om*T(end-(P*Nt)+1:end)) cos(2*2*pi*Om*T(end-(P*Nt)+1:end)) sin(2*2*pi*Om*T(end-(P*Nt)+1:end))];
        TH = YR\Y(end-(P*Nt)+1:end);        
        a_rms_R(i,j) = sqrt(TH(1)^2 + 0.5*sum(TH(2:end).^2));
        ph_rs_R(i,j) = rad2deg(angle((TH(2)-1j*TH(3))./Ufint(2)));
        
        % INTERPOLATION APPROACH
        % Interpolating response from grid T that is sampled at the fixed
        % sampling rate "fs" to grid Tint that is sampled at the sampling
        % rate Ntint*Om
        Pint = floor(T(end)*Om);
        Tint = linspace(0, Pint/Om, Ntint*Pint+1)';  Tint(end) = [];
        Yint = interp1(T, Y, Tint, 'pchip');

        % Removal of transients on interpolated response data
        Yint = Yint(end-(P*Ntint)+1:end);
        y(:, i) = Yint;
        
        Yrint = mean(reshape(Yint, Ntint, P), 2);
        Yfint = fft(Yrint)/(Ntint/2); Yfint(1) = Yfint(1)/2;
        
        a_rms(i,j) = sqrt(Yfint(1)^2 + 0.5*vecnorm(Yfint(2:Ntint/2)).^2);
        ph_rs(i,j) = rad2deg(angle(Yfint(2)/Ufint(2)));
%         ph_rs(i,j) = rad2deg(angle(Yfint(2)));
        if sum(any(isnan(Y)))
            fprintf('Error: simulation exploded. Try increase fs. r:%d\n',i)
            break % don't quit, we still want to save data.
        end
    end
    j = j+1;
    fprintf('A: %g\n',A)
end

%% Plotting
fg1 = 10;
fg2 = 20;

figure(fg1)
clf()

figure(fg2)
clf()
colos = distinguishable_colors(length(exc_lev));
aa = gobjects(size(exc_lev));
for iex=1:length(exc_lev)
    figure(fg1)
    plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-', 'Color', colos(iex,:)); hold on
%     plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2), '.--', 'Color', colos(iex,:))
    plot(Om_vec, a_rms_R(:,iex), 'x-', 'Color', colos(iex,:))
    
    figure(fg2)
    aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3), '-', 'Color', colos(iex,:)); hold on
%     plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3), '.--', 'Color', colos(iex,:))
    plot(Om_vec, ph_rs_R(:,iex), 'x-', 'Color', colos(iex,:))
    legend(aa(iex), sprintf('F = %.2f', exc_lev(iex)));
end

figure(fg1)
set(gca,'Yscale', 'log')
xlim(sort([Om_s Om_e])/2/pi)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('RMS response amplitude (m)')
% savefig(sprintf('./fig/pnlssfrf_A%d_Amp.fig',Alevel))
% print(sprintf('./fig/TMdssex_frf_Amp_fs%d.eps',fs), '-depsc')

figure(fg2)
xlim(sort([Om_s Om_e])/2/pi)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
legend(aa(1:end), 'Location', 'northeast')
% savefig(sprintf('./fig/pnlssfrf_A%d_Phase.fig',Alevel))
% print(sprintf('./fig/TMdssex_frf_Phase_fs%d.eps',fs), '-depsc')