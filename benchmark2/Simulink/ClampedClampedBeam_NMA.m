%========================================================================
% DESCRIPTION: 
% Investigation of the dynamics of a duffing oscillator
% nonlinearity using NLvib and simulated measurements of the backbone
%========================================================================

clearvars;
close all;
clc;

srcpath = '../../src/nlvib';
addpath(genpath(srcpath));
srcpath = '../../src/simulink';
addpath(genpath(srcpath));
srcpath = '../';
addpath(genpath(srcpath));

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

%% Define system

% Fundamental parameters
Dmod = [.38 .12 .12 .08 .08]*.01;
Nmod = 5;
setup = 'New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L_2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
load(['beam_New_Design_Steel_analytical_5t_' num2str(thickness*1000) 'mm.mat'])
model.b=b;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(:).*om(:));
K = diag(om.^2);

% polynomial terms
p_quad = zeros(sum(1:Nmod),Nmod);
p_cub = zeros(sum(cumsum(1:Nmod)),Nmod);
ctr_quad = 1; ctr_cub = 1;

for jj = 1:Nmod
    for kk = jj:Nmod
        % quadratic terms
        p_quad(ctr_quad,jj) = p_quad(ctr_quad,jj)+1;
        p_quad(ctr_quad,kk) = p_quad(ctr_quad,kk)+1;
        ctr_quad = ctr_quad+1;
        for ll = kk:Nmod
            % cubic terms
            p_cub(ctr_cub,jj) = p_cub(ctr_cub,jj)+1;
            p_cub(ctr_cub,kk) = p_cub(ctr_cub,kk)+1;
            p_cub(ctr_cub,ll) = p_cub(ctr_cub,ll)+1;
            ctr_cub = ctr_cub+1;
        end
    end
end

p = [p_cub];

% coefficients
E=zeros(sum(cumsum(1:Nmod)),Nmod);

for rr = 1:Nmod
    ctr = 1;
%         for jj = 1:Nmod
%             for kk = jj:Nmod
%                 % quadratic coeffs
%                 E(ctr,rr) = model.a(jj,kk,rr);
%                 ctr = ctr+1;
%             end
%         end
%         ctr = 1;
    for jj = 1:Nmod
        for kk = jj:Nmod
            for ll = kk:Nmod
                % cubic coeffs
                E(ctr,rr) = model.b(jj,kk,ll,rr);
                ctr = ctr+1;
            end
        end
    end
end

% Fundamental harmonic of external forcing
Fex1 = PHI_L_2;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);

% Number of degrees of freedom
n = oscillator.n;

%% Linear modal analysis
[PHI_lin,OM2] = eig(oscillator.K,oscillator.M);
[om_lin,ind] = sort(sqrt(diag(OM2)));
PHI_lin = PHI_lin(:,ind);

%% NMA
imod = 1;           % mode to be analyzed
switch imod
    case 1
        H = 9;
        log10a_s = -5;    % start vibration level (log10 of modal mass)
        log10a_e = -2;       % end vibration level (log10 of modal mass)
    case 3
        H = 3;
        log10a_s = -7;    % start vibration level (log10 of modal mass)
        log10a_e = -3;       % end vibration level (log10 of modal mass)
    case 5
        H = 1;
        log10a_s = -9;    % start vibration level (log10 of modal mass)
        log10a_e = -4;       % end vibration level (log10 of modal mass)
end

N=2*3*H+1;
analysis = 'NMA';
inorm = 1;          % coordinate for phase normalization

% Initial guess vector y0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
om = om_lin(imod); phi = PHI_lin(:,imod);
Psi = zeros((2*H+1)*Nmod,1);
Psi(Nmod+(1:Nmod)) = phi;
x0 = [Psi;om;0];

ds      = .1;
Sopt    = struct('Dscale',[1e-6*ones(size(x0,1)-2,1);1;1e-1;1],...
    'dynamicDscale',1,'stepmax',5e4);
[X_NM,Solinfo,Sol] = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,H,N,analysis,inorm),...
    log10a_s,log10a_e,ds, Sopt);

Psi_HB = X_NM(1:end-3,:);
om_HB = X_NM(end-2,:);
del_HB = X_NM(end-1,:);
log10a_NMA = X_NM(end,:);
a_NMA = 10.^log10a_NMA;
Q_HB = Psi_HB.*repmat(a_NMA,size(Psi_HB,1),1);
% fundamental harmonic motion
Y_HB_1 = Q_HB(n+(1:n),:)-1i*Q_HB(2*n+(1:n),:);
Y_HB_1 = Y_HB_1'*PHI_L_2';

% Define amplitude as magnitude of the fundamental harmonic of the
% first coordinate's displacement
w_L_2_NMA_sum = zeros(2*H+1,length(om_HB));
for k = 1:n
    Qc{k} = [Q_HB(k,:);Q_HB(n+k:n:end,:)];
    w_L_2_NMA{k} = PHI_L_2(k)*Qc{k};                  % get displacement at center caused by each mode in time domain
    w_L_2_NMA_sum = [w_L_2_NMA_sum + w_L_2_NMA{k}];
end

a_w_L_2_NMA = sqrt([1 0.5*ones(1,2*H)]*w_L_2_NMA_sum.^2); % compute amplitude

%% Setup simulated experiments

Shaker = 'no'; % 'yes', 'no'
phase_lag = 90; % phase lag in degree

x0beam = 0; % intial condition beam integrator
x0vco = 0; % initial condition VCO integrator

% PLL controller
omega_0 = om_lin(imod); % center frequency
a = 0.02*2*pi; % low pass filter

% state-space model
A = [zeros(n,n), eye(n);
  -oscillator.M\oscillator.K, -oscillator.M\oscillator.D];

% localize nonlinearity
loc_nl = 1;
loc_exc = Fex1;

% input matrices
B = [zeros(n,1);oscillator.M\loc_exc];
% input matrix for nonlinear force
NL_coeff = oscillator.nonlinear_elements{1}.coefficients;
B_nl = [zeros(n,1);oscillator.M\loc_nl]*NL_coeff;

% localization matrices
T_nl = PHI_L_2*[eye(n,n) zeros(n,n)];

T_exc = zeros(1,2*n);
T_exc(1) = PHI_L_2;

T_disp = PHI_L_2*[eye(n,n) zeros(n,n)];

%% shaker model

% source: Master thesis Florian Morlock, INM, University of Stuttgart

%%%%% mechanical parameters
M_T = 0.0243; % Table Masse [kg]
M_C = 0.0190; % Coil Masse [kg]
K_C = 8.4222*10^7; % spring constant Coil-Table [N/m]

K_S = 20707; % spring constant Table-Ground [N/m]
C_C = 57.1692; % damping constant Coil-Table [Ns/m]
C_S = 28.3258; % damping constant Table-Ground [Ns/m]

%%%%% electrical
L = 140*10^-6; % inuctivity [H]
R = 3.00; % resistance [Ohm]
Tau = 15.4791; % shaker ratio of thrust to coil current [N/A]

%%%%% State space model
A_shaker = [- R/L 0 -Tau/L 0 0; 0 0 1 0 0; ...
    Tau/M_C -K_C/M_C -C_C/M_C K_C/M_C C_C/M_C; 0 0 0 0 1; ...
    0 K_C/M_T C_C/M_T -(K_C+K_S)/M_T -(C_C+C_S)/M_T];
B_shaker = [Tau/M_C 0 0 0 0; 0 0 0 0 1/M_T]';
C_shaker = [0 0 0 1 0; 0 0 0 0 1];
D_shaker = [0 0 0 0];

% stunger parameters
E_stinger = 210000 ;%in N/mm^2
A_stinger = pi*2^2; % in mm
l_stinger = 0.0200; %in m
k_stinger = (E_stinger*A_stinger)/l_stinger;

%% switch Shaker
switch Shaker
    case 'no' % without Shaker
        
        % PLL controller
        P = 0; % proportional gain
        I = 50; % integrator gain
        D = 0; % derivative gain
        
        time_interval = [0.2 10 0.2 20 0.2 30 0.2 40 0.2 50 0.2 50];
        simin.time = zeros(1,13);
        for i = 1:12
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);  % Simulation time in seconds
        simin.signals.values = 0.005*[0 3 3 10 10 25 25 60 60 100 100 160 160]';
        simin.signals.dimensions = 1;
        
        % simulation of experiment
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
        sim('Duffing_without_Shaker')
        disp('Simulation of experiment succeeded')
        
    case 'yes' % with Shaker
        
        % PLL controller
        P = 10; % proportional gain
        I = 50; % integrator gain
        D = 0.01; % derivative gain
        
        time_interval = [0.1 5 0.1 15 0.1 15 0.1 15 0.1 15 0.1 15];
        simin.time = zeros(1,13);
        for i = 1:12
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);
        simin.signals.values = 30*[0 5 5 20 20 40 40 100 100 180 180 270 270]';
        simin.signals.dimensions = 1;
        
        % simulation of experiment
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
        sim('Duffing_shaker')
        disp('Simulation of experiment succeeded')
end

simulation.disp = displacement.signals.values;
simulation.tvals = displacement.time;
simulation.Fvals = excitation_force.signals.values;
simulation.freqvals = exc_freq.signals.values;

simulation.Signalbuilder = simin.time';

%% Analysis of simualted measurements
opt.NMA.exc_DOF = exc_node; % index of drive point
opt.NMA.Fs = 5000; % sample frequency in Hz
opt.NMA.var_step = 1; % 0 for constant step size, 1 for variable step size
opt.NMA.periods = 50; % number of analyzed periods

opt.NMA.n_harm = 10; % number of harmonics considered
opt.NMA.min_harm_level = 0.015; % minimal level relative to highest peak
opt.NMA.eval_DOF = exc_node; % DOF for analysis

% number of modes considered in linear EMA
modes_considered = 1;
linear_EMA_sensors = 1;

% results linear modal analysis
res_LMA.freq = om_lin(modes_considered)/2/pi;
res_LMA.damp = D(modes_considered);
res_LMA.Phi = PHI_lin(linear_EMA_sensors,modes_considered);
res_LMA.modes_considered = modes_considered;

% results nonlinear modal analysis
res_bb = signal_processing_backbone_simulation(simulation, opt.NMA);
res_damp = nonlinear_damping( res_LMA, res_bb);
names = [fieldnames(res_bb); fieldnames(res_damp)];
res_NMA = cell2struct([struct2cell(res_bb); struct2cell(res_damp)], names, 1);

% Compare modal characteristics for experiment and Harmonic Balance methods

% Modal frequency vs. amplitude
figure;
semilogx(abs(Y_HB_1),om_HB/om_lin(imod),'g-', 'LineWidth', 2);
hold on
semilogx(abs(res_NMA.Psi_tilde_i(opt.NMA.eval_DOF,:)),res_NMA.om_i/(res_LMA.freq(1)*2*pi),'k.','MarkerSize',10)
xlabel('amplitude in m'); ylabel('$\omega/\omega_0$')
legend('NMA with NLvib','simulated experiment')

% Modal damping ratio vs. amplitude
figure; 
semilogx(abs(Y_HB_1),del_HB*1e2,'g-', 'LineWidth', 2);
hold on
semilogx(abs(res_NMA.Psi_tilde_i(opt.NMA.eval_DOF,:)),abs(res_NMA.del_i_nl)*1e2,'k.','MarkerSize',10)
xlabel('amplitude in m'); ylabel('modal damping ratio in %')
legend('NMA with NLvib','simulated experiment')
