%========================================================================
% DESCRIPTION: 
% Investigation of the dynamics of a beam with elastic unilateral spring
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

imod = 1;  % Desired mode
Shaker = 'no'; % 'yes', 'no'

%% System definition
len = 0.70;
hgt = 0.03;
thk = hgt;
E   = 185e9;
rho = 7830.0;
BCs = 'clamped-free';

Nn = 8;
beam = FE_EulerBernoulliBeam(len, hgt, thk, E, rho, BCs, Nn);
fdof = beam.n-1;
Fex1 = zeros(beam.n, 1);  Fex1(fdof) = 1;
% Apply unilateral spring element at node 4 in translational direction
Nnl = 4; % index includes clamped node
dir = 'trans';
kn  = 1.3e6;
gap = 1e-3;
add_nonlinear_attachment(beam, Nnl, dir, 'unilateralspring', ...
    'stiffness', kn, 'gap', gap);
Nd = size(beam.M, 1);
n = Nd;

%% Modal analysis the linearized system

% Modes for free sliding contact
[PHI_free,OM2] = eig(beam.K, beam.M);
om_free = sqrt(diag(OM2));
% Sorting
[om_free,ind] = sort(om_free);
PHI_free = PHI_free(:,ind);

% Modes for fixed contact
K_ex = eye(length(beam.M));
inl = find(beam.nonlinear_elements{1}.force_direction);
K_ex(inl,inl) = kn;
K_ex = beam.K + K_ex;
[PHI_fixed,OM2] = eig(K_ex,beam.M);
om_fixed = sqrt(diag(OM2));
% Sorting
[om_fixed,ind] = sort(om_fixed);
PHI_fixed = PHI_fixed(:,ind);

%% Define linear modal damping

% desired modal damping ratios for first two modes
D1 = 0.008;
D2 = 0.008;

% define Rayleigh damping based on modal damping ratios
beta = 2*(D2*om_fixed(2)-om_fixed(1)*D1)/(om_fixed(2)^2-om_fixed(1)^2);
alpha = 2*om_fixed(1)*D1 - beta*om_fixed(1)^2;

beam.D = alpha*beam.M + beta*K_ex;

% mass-normalized mode shapes
qq =diag(PHI_fixed'*beam.M*PHI_fixed);
PHI_fixed = PHI_fixed./repmat(sqrt(qq)',n,1);

cc = diag(PHI_fixed'*beam.D*PHI_fixed);
D = cc./(2*om_fixed); % modal damping ratios

%% Nonlinear modal analysis using harmonic balance

Nt = 2^10;

switch imod
    case 1
        Nh = 9;      
        log10a_s = -4;
        log10a_e = -0.5;
        dl10a = 0.01;
        dl10amax = 0.05;
    case 2
        Nh = 9;
        log10a_s = -4.5;
        log10a_e = -0.5;
        dl10a = 0.005;
        dl10amax = 0.025;
    case 3
        Nh = 3;
        log10a_s = -4.5;
        log10a_e = -0.5;
        dl10a = 0.0001;
        dl10amax = 0.005;
end

Nhc = 2*Nh+1;
Dscale = [1e-1*ones(Nd*Nhc,1); om_fixed(imod); D(imod); 1.0];
        
inorm = Nd-1;

X0 = zeros(Nd*Nhc+2, 1);
X0(Nd+(1:Nd)) = PHI_fixed(:,imod);
X0(end-1) = om_fixed(imod);
X0(end) = D(imod);

beam.Fex1 = beam.Fex1*0;

Sopt = struct('jac', 'full', 'Dscale', Dscale, 'dynamicDscale', 1, ...
    'dsmax', dl10amax);
Sopt = struct('jac', 'full', 'dsmax', dl10amax, 'dynamicDscale', 1);

fscl = mean(abs(beam.K*PHI_fixed(:,imod)));
Xbb = solve_and_continue(X0, ...
    @(X) HB_residual(X, beam, Nh, Nt, 'nma', inorm, fscl), ...
    log10a_s, log10a_e, dl10a, Sopt);
Bkb = [10.^Xbb(end,:);  % modal amplitude
    Xbb(end-2,:);  % frequency
    Xbb(end-1,:);  % damping factor
    atan2d(-Xbb(2*Nd+fdof,:), Xbb(Nd+fdof,:)); % phase
    (10.^Xbb(end,:)).*sqrt([1 0.5*ones(1, 2*Nh)]*Xbb(fdof:Nd:end-3,:).^2)]';

% Interpret solver output
Psi_HB = Xbb(1:end-3,:);
om_HB = Xbb(end-2,:);
del_HB = Xbb(end-1,:);
log10a_HB = Xbb(end,:);
a_HB = 10.^log10a_HB;
Q_HB = Psi_HB.*repmat(a_HB,size(Psi_HB,1),1);
% fundamental harmonic motion
Y_HB_1 = Q_HB(n+(1:n),:)-1i*Q_HB(2*n+(1:n),:);

%% Setup simulated experiments

exc_node = 8; % node for external excitation
phase_lag = 90; % phase lag in degree

x0beam = 0; % intial condition beam integrator
x0vco = 0; % initial condition VCO integrator

% PLL controller
omega_0 = om_free(imod); % center frequency
a = 0.02*2*pi; % low pass filter

% state-space model
A = [zeros(n,n), eye(n);
  -beam.M\beam.K, -beam.M\beam.D];

% localize nonlinearity
loc_nl = beam.nonlinear_elements{1}.force_direction;
loc_exc = zeros(n,1);
loc_exc(2*(exc_node-1-1)+1) = 1;

% input matrices
B = [zeros(n,1);beam.M\loc_exc];
% input matrix for nonlinear force
B_nl = [zeros(n,1);beam.M\loc_nl];

% localization matrices
T_nl = zeros(1,2*n);
T_nl(1:n) = beam.nonlinear_elements{1}.force_direction';

T_exc = zeros(1,2*n);
T_exc(2*(exc_node-1-1)+1) = 1;

T_disp = [eye(n,n) zeros(n,n)];

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

%% simulation of experiment

switch Shaker
    case 'yes'
        time_interval = [0.1 7 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5];
        simin.time = zeros(1,length(time_interval)+1);
        for i = 1:length(time_interval)
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end); % Simulation time in seconds
        
        P = 25; % proportional gain
        I = 50; % integrator gain
        Der = 0.05; % Derivative gain
        
        switch imod
            case 1
                simin.signals.values = 10*[0 1 1 5 5 15 15 30 30 50 50 90 90 150 150 300 300]';
            case 2
                simin.signals.values = 50*[0 1 1 5 5 15 15 20 20 30 30 50 50 80 80 150 150]';
            case 3
                simin.signals.values = 300*[0 5 5 15 15 30 30 50 50 80 80 120 120 160 160 220 220]';
        end
        simin.signals.dimensions = 1;
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
        sim('unilateral_shaker')
        disp('Simulation of experiment succeeded')
        
    case 'no'
        time_interval = [0.1 10 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5 0.1 5];
        simin.time = zeros(1,length(time_interval)+1);
        for i = 1:length(time_interval)
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);
        
        P = 25; % proportional gain
        I = 50; % integrator gain
        Der = 0; % Derivative gain
        
        switch imod
            case 1
                simin.signals.values = 5*[0 0.1 0.1 0.5 0.5 1.5 1.5 3 3 5 5 8 8 15 15 25 25 50 50]';
            case 2
                simin.signals.values = 5*[0 1 1 5 5 15 15 30 30 50 50 90 90 150 150 350 350 700 700]';
            case 3
                simin.signals.values = 100*[0 3 3 20 20 35 35 50 50 70 70 100 100 150 150 500 500 1000 1000]';
        end
        simin.signals.dimensions = 1;
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
        sim('unilateral_no_shaker')
        disp('Simulation of experiment succeeded')
end

simulation.disp = displacement.signals.values(:,1:2:end);
simulation.tvals = displacement.time;
simulation.Fvals = excitation_force.signals.values;
simulation.freqvals = exc_freq.signals.values;

simulation.Signalbuilder = simin.time;

%% Analysis of simualted measurements

opt.NMA.exc_DOF = exc_node; % index of drive point
opt.NMA.var_step = 1; % 0 for constant step size, 1 for variable step size
switch imod
    case 1
        opt.NMA.Fs = 5000; % sample frequency in Hz
        opt.NMA.periods = 100; % number of analyzed periods
    case 2
        opt.NMA.Fs = 5000; % sample frequency in Hz
        opt.NMA.periods = 350; % number of analyzed periods
    case 3
        opt.NMA.Fs = 50000; % sample frequency in Hz
        opt.NMA.periods = 500; % number of analyzed periods
end

opt.NMA.n_harm = 20; % number of harmonics considered
opt.NMA.min_harm_level = 0.015; % minimal level relative to highest peak
opt.NMA.eval_DOF = exc_node-1; % DOF for analysis

% number of modes considered in linear EMA
modes_considered = 1:7;
linear_EMA_sensors = 1:2:n; % only "measure" in translational direction

% results linear modal analysis
res_LMA.freq = om_free(modes_considered)/2/pi;
res_LMA.damp = D(modes_considered);
res_LMA.Phi = PHI_free(linear_EMA_sensors,modes_considered);
res_LMA.modes_considered = modes_considered;

%% results nonlinear modal analysis
res_bb = signal_processing_backbone_simulation(simulation, opt.NMA);
%%
res_damp = nonlinear_damping( res_LMA, res_bb);
names = [fieldnames(res_bb); fieldnames(res_damp)];
res_NMA = cell2struct([struct2cell(res_bb); struct2cell(res_damp)], names, 1);

% Compare modal characteristics for experiment and Harmonic Balance methods

% Modal frequency vs. amplitude
figure;
semilogx(abs(Y_HB_1(2*(exc_node-1-1)+1,:)),om_HB/om_free(imod),'g-', 'LineWidth', 2);
hold on
semilogx(abs(res_NMA.Psi_tilde_i(opt.NMA.eval_DOF,:)),res_NMA.om_i/(res_LMA.freq(imod)*2*pi),'k.','MarkerSize',10)
xlabel('amplitude in m'); ylabel('$\omega/\omega_0$')
legend('NMA with NLvib','simulated experiment')

% Modal damping ratio vs. amplitude
figure; 
semilogx(abs(Y_HB_1(2*(exc_node-1-1)+1,:)),del_HB*1e2,'g-', 'LineWidth', 2);
hold on
semilogx(abs(res_NMA.Psi_tilde_i(opt.NMA.eval_DOF,:)),abs(res_NMA.del_i_nl)*100,'k.','MarkerSize',10)
xlabel('amplitude in m'); ylabel('modal damping ratio in %')
legend('NMA with NLvib','simulated experiment')
