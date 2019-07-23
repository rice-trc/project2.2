%========================================================================
% DESCRIPTION: 
% Investigation of the dynamics of a beam with elastic dry friction
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

%% Define system (undamped)

% Properties of the beam
len = 0.7;                % length in m
height = .03;       % height in the bending direction in m
thickness = height;   % thickness in the third dimension in m
E = 185e9;              % Young's modulus in Pa
rho = 7830;             % density in kg/m^3
BCs = 'clamped-free';   % constraints

% Setup one-dimensional finite element model of an Euler-Bernoulli beam
n_nodes = 8;            % number of equidistant nodes along length
beam = FE_EulerBernoulliBeam(len,height,thickness,E,rho,...
    BCs,n_nodes);
n = beam.n;

% Apply elastic Coulomb element at node 4 in translational direction
nl_node = 4; % index includes clamped node
dir = 'trans';
kt = 1.3e6; % tangential stiffness in N/m
muN = 1;    % friction limit force in N
add_nonlinear_attachment(beam,nl_node,dir,'elasticdryfriction',...
    'stiffness',kt,'friction_limit_force',muN, 'ishysteretic', 1);

%% Modal analysis the linearized system

% Modes for free sliding contact
[PHI_free,OM2] = eig(beam.K,beam.M);
om_free = sqrt(diag(OM2));
% Sorting
[om_free,ind] = sort(om_free);
PHI_free = PHI_free(:,ind);

% Modes for fixed contact
K_ex = eye(length(beam.M));
inl = find(beam.nonlinear_elements{1}.force_direction);
K_ex(inl,inl) = kt;
K_ex = beam.K + K_ex;
[PHI_fixed,OM2] = eig(K_ex,beam.M);
om_fixed = sqrt(diag(OM2));
% Sorting
[om_fixed,ind] = sort(om_fixed);
PHI_fixed = PHI_fixed(:,ind);

%% Define linear modal damping

% desired modal damping ratios for first two modes
D1 = 0.008;
D2 = 0.002;

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
analysis = 'NMA';

imod = 3;           % mode to be analyzed
Ntd = 2^10;         % number of time samples per period
inorm = 2;          % coordinate for phase normalization

% Analysis parameters
switch imod
    case 1
        H = 13;             % harmonic order
        log10a_s = -5.7;    % start vibration level (log10 of modal mass)
        log10a_e = -3;      % end vibration level (log10 of modal mass)
        ds = .02;
    case 2
        H = 7;             % harmonic order
        log10a_s = -7;    % start vibration level (log10 of modal mass)
        log10a_e = -4;      % end vibration level (log10 of modal mass)
        ds = .02;
    case 3
        H = 1;             % harmonic order
        log10a_s = -7;    % start vibration level (log10 of modal mass)
        log10a_e = -4;      % end vibration level (log10 of modal mass)
        ds = .01;
end

% Initial guess vector x0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
om = om_fixed(imod); phi = PHI_fixed(:,imod);
Psi = zeros((2*H+1)*n,1);
Psi(n+(1:n)) = phi;
x0 = [Psi;om;D(1)];

% Solve and continue w.r.t. Om
fscl = mean(abs(beam.K*phi));
Sopt = struct('stepmax',2000);
X_HB = solve_and_continue(x0,...
    @(X) HB_residual(X,beam,H,Ntd,analysis,inorm,fscl),...
    log10a_s,log10a_e,ds,Sopt);

% Interpret solver output
Psi_HB = X_HB(1:end-3,:);
om_HB = X_HB(end-2,:);
del_HB = X_HB(end-1,:);
log10a_HB = X_HB(end,:);
a_HB = 10.^log10a_HB;
Q_HB = Psi_HB.*repmat(a_HB,size(Psi_HB,1),1);
% fundamental harmonic motion
Y_HB_1 = Q_HB(n+(1:n),:)-1i*Q_HB(2*n+(1:n),:);

%% Setup simulated experiments

Shaker = 'no'; % 'yes', 'no'

exc_node = 8; % node for external excitation
simtime = 30;   % Simulation time in seconds
phase_lag = 90; % phase lag in degree

x0beam = 0; % intial condition beam integrator
x0vco = 0; % initial condition VCO integrator

% PLL controller
omega_0 = om_fixed(imod); % center frequency
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
        time_interval = [0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4];
        simin.time = zeros(1,13);
        for i = 1:12
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);
        switch imod
            case 1
                simin.signals.values = [0 0.2 0.2 0.5 0.5 0.75 0.75 1 1 2 2 3.5 3.5]';
            case 2
                simin.signals.values = [0 0.2 0.2 0.5 0.5 0.75 0.75 1 1 2 2 3.5 3.5]';
            case 3
                simin.signals.values = 10*[0 0.2 0.2 0.5 0.5 0.75 0.75 1 1 2 2 3.5 3.5]'; 
        end
        simin.signals.dimensions = 1;
        
        P = 5; % proportional gain
        I = 50; % integrator gain
        Der = 0; % Derivative gain
        
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
        sim('DryElasticFriction_voltage')
        disp('Simulation of experiment succeeded')
        
    case 'no'
        time_interval = [0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4];
        simin.time = zeros(1,13);
        for i = 1:12
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);
        switch imod
            case 1
                simin.signals.values = [0 0.05 0.05 0.1 0.1 0.25 0.25 0.3 0.3 0.4 0.4 0.6 0.6]';
            case 2
                simin.signals.values = [0 0.01 0.01 0.1 0.1 0.25 0.25 0.4 0.4 0.8 0.8 1.2 1.2]';
            case 3
                simin.signals.values = 5*[0 0.05 0.05 0.1 0.1 0.25 0.25 0.3 0.3 0.4 0.4 0.6 0.6]';
        end
        simin.signals.dimensions = 1;
        
        P = 5; % proportional gain
        I = 50; % integrator gain
        Der = 0; % Derivative gain
        
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
        sim('DryElasticFriction_Force')
        disp('Simulation of experiment succeeded')
end

simulation.disp = displacement.signals.values(:,1:2:end);
simulation.tvals = displacement.time;
simulation.Fvals = excitation_force.signals.values;
simulation.freqvals = exc_freq.signals.values;

simulation.Signalbuilder = simin.time;


%% Analysis of simualted measurements

opt.NMA.exc_DOF = exc_node; % index of drive point
opt.NMA.Fs = 30000; % sample frequency in Hz
opt.NMA.var_step = 1; % 0 for constant step size, 1 for variable step size
opt.NMA.periods = 350; % number of analyzed periods

opt.NMA.n_harm = 10; % number of harmonics considered
opt.NMA.min_harm_level = 0.015; % minimal level relative to highest peak
opt.NMA.eval_DOF = exc_node-1; % DOF for analysis

% number of modes considered in linear EMA
modes_considered = 1:5;
linear_EMA_sensors = 1:2:n; % only "measure" in translational direction

% results linear modal analysis
res_LMA.freq = om_fixed(modes_considered)/2/pi;
res_LMA.damp = D(modes_considered);
res_LMA.Phi = PHI_fixed(linear_EMA_sensors,modes_considered);
res_LMA.modes_considered = modes_considered;

% results nonlinear modal analysis
res_bb = signal_processing_backbone_simulation(simulation, opt.NMA);
res_damp = nonlinear_damping( res_LMA, res_bb);
names = [fieldnames(res_bb); fieldnames(res_damp)];
res_NMA = cell2struct([struct2cell(res_bb); struct2cell(res_damp)], names, 1);

% Compare modal characteristics for experiment and Harmonic Balance methods

% Modal frequency vs. amplitude
figure;
semilogx(abs(Y_HB_1(2*(exc_node-1-1)+1,:)),om_HB/om_fixed(imod),'g-', 'LineWidth', 2);
hold on
semilogx(abs(res_NMA.Psi_tilde_i(opt.NMA.eval_DOF,:)),res_NMA.om_i/(res_LMA.freq(imod)*2*pi),'k.','MarkerSize',10)
xlabel('amplitude in m'); ylabel('\omega/\omega_0')
legend('NMA with NLvib','simulated experiment')

% Modal damping ratio vs. amplitude
figure; 
semilogx(abs(Y_HB_1(2*(exc_node-1-1)+1,:)),del_HB*1e2,'g-', 'LineWidth', 2);
hold on
semilogx(abs(res_NMA.Psi_tilde_i(opt.NMA.eval_DOF,:)),abs(res_NMA.del_i_nl)*100,'k.','MarkerSize',10)
xlabel('amplitude in m'); ylabel('modal damping ratio in %')
legend('NMA with NLvib','simulated experiment')
