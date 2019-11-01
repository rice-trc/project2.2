%========================================================================
% DESCRIPTION: 
% Investigation of the dynamics of a beam with elastic dry friction
% nonlinearity using NLvib and simulated measurements of the backbone
%========================================================================

clearvars;
close all;
clc;

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

addpath('../../src/matlab/')
addpath('../../src/nlvib/SRC/')
addpath('../../src/nlvib/SRC/MechanicalSystems/')
addpath('../../src/simulink/')

imod = 1;           % mode to be analyzed
Shaker = 'no';      % 'yes', 'no'
valorest = 'est';    % 'val', 'est'
fdata = 1;  % full data (no selection of periodic portions and the rest)

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
nl_node = 4; % i                           ndex includes clamped node
dir = 'trans';
kt = 1.3e6; % tangential stiffness in N/m
muN = 1;    % friction limit force in N
add_nonlinear_attachment(beam,nl_node,dir,'elasticdryfriction',...
    'stiffness',kt,'friction_limit_force',muN, ...
    'ishysteretic', true);

%% Modal analysis the linearized system

% Modes for free sliding contact
[PHI_free,OM2] = eig(beam.K,beam.M);
om_free = sqrt(diag(OM2));
% Sorting
[om_free,ind] = sort(om_free);
PHI_free = PHI_free(:,ind);

% Modes for fixed contact
K_ex = zeros(length(beam.M));
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
D1 = 8e-3;
D2 = 2e-3;

% define Rayleigh damping based on modal damping ratios
beta = 2*(D2*om_fixed(2)-om_fixed(1)*D1)/(om_fixed(2)^2-om_fixed(1)^2);
alpha = 2*om_fixed(1)*D1 - beta*om_fixed(1)^2;

beam.D = alpha*beam.M + beta*K_ex;

% mass-normalized mode shapes
qq =diag(PHI_fixed'*beam.M*PHI_fixed);
PHI_fixed = PHI_fixed./repmat(sqrt(qq)',n,1);

cc = diag(PHI_fixed'*beam.D*PHI_fixed);
D = cc./(2*om_fixed); % modal damping ratios

%% Setup simulated experiments

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
sc = 1.0;
switch valorest
    case 'est'
        sc = 1.0;
    case 'val'
        sc = 0.9;
end

switch Shaker
    case 'yes'
        time_interval = [0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4];
        simin.time = zeros(1,length(time_interval)+1);
        for i = 1:length(time_interval)
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);
        switch imod
            case 1
                simin.signals.values = sc*[0 0.02 0.02 0.05 0.05 0.1 0.1 0.2 0.2 0.4 0.4 0.5 0.5 0.6 0.6 0.7 0.7 0.8 0.8 1 1 1.2 1.2 1.3 1.3 1.5 1.5 1.8 1.8]';
            case 2
                simin.signals.values = sc*[0 0.2 0.2 0.3 0.3 0.4 0.4 0.5 0.5 0.6 0.6 0.7 0.7 0.8 0.8 1 1 2 2 3.5 3.5]';
            case 3
                simin.signals.values = sc*10*[0 0.2 0.2 0.3 0.3 0.4 0.4 0.5 0.5 0.6 0.6 0.75 0.75 1 1 1.5 1.5 2 2 3.5 3.5]'; 
        end
        simin.signals.dimensions = 1;
        
        P = 5; % proportional gain
        I = 50; % integrator gain
        Der = 0; % Derivative gain
        
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
%         sim('DryElasticFriction_Shaker')
        sim('./Simulink/DryElasticFriction_voltage')
        disp('Simulation of experiment succeeded')
        
    case 'no'
        time_interval = [0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4 0.1 4];
        simin.time = zeros(1,length(time_interval)+1);
        for i = 1:length(time_interval)
            simin.time(i+1) = simin.time(i)+time_interval(i);
        end
        simtime = simin.time(end);
        switch imod
            case 1
                simin.signals.values = sc*[0 0.001 0.001 0.002 0.002 0.003 0.003 0.02 0.02 0.05 0.05 0.08 0.08 0.1 0.1 0.13 0.13 0.17 0.17 0.21 0.21 0.23 0.23 0.25 0.25 0.28 0.28 0.3 0.3]';
            case 2
                simin.signals.values = sc*[0 0.01 0.01 0.15 0.15 0.25 0.25 0.35 0.35 0.5 0.5 0.8 0.8 1 1 1.2 1.2 2 2 3 3]';
            case 3
                simin.signals.values = sc*5*[0 0.05 0.05 0.07 0.07 0.1 0.1 0.15 0.15 0.2 0.2 0.25 0.25 0.3 0.3 0.35 0.35 0.4 0.4 0.6 0.6]';
        end
        simin.signals.dimensions = 1;
        
        P = 5; % proportional gain
        I = 50; % integrator gain
        Der = 0; % Derivative gain
        
        disp('---------------------------------------------------')
        disp('Simulation of experiment started')
%         sim('DryElasticFriction_No_Shaker')
        sim('./Simulink/DryElasticFriction_Force')
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
opt.NMA.Fs = 5000; % sample frequency in Hz
opt.NMA.periods = 100; % number of analyzed periods
% opt.NMA.Fs = 30000; % sample frequency in Hz
% opt.NMA.periods = 350; % number of analyzed periods

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

%% Process for NM-ROM
fex1 = zeros(size(res_damp.Phi_tilde_i,1),1);  fex1(exc_node-1) = 1;

a = res_NMA.q_i;
p2 = (res_NMA.om_i.^2-2*(res_NMA.del_i_nl.*res_NMA.om_i).^2)';
om4 = res_NMA.om_i'.^4;
Phi = res_NMA.Phi_tilde_i;
Fsc = abs((abs(res_NMA.Phi_tilde_i'*fex1)./a)).^2;
mA = abs(res_NMA.Psi_tilde_i(exc_node-1,:))*sqrt(2);

save(['Data/SimExp_shaker_' Shaker '_NMROM.mat'], 'res_NMA', 'a', 'p2', 'om4', 'Phi', 'Fsc', 'mA')

%% Process for pnlss
if fdata~=1
    PNLSS = opt.NMA;
    PNLSS.periods = 20;
    PNLSS.ppr = 1;
    [t,u,y] = pnlss_preparing_backbone_simulation(simulation, PNLSS);
    fdof = 'exc_node';
    fsamp = PNLSS.Fs;
    save(['Data/SimExp_shaker_' Shaker '_' valorest '.mat'], 't','u','y','fsamp','PNLSS');
else
    PNLSS = opt.NMA;
    PNLSS.periods = 20;
    PNLSS.ppr = 1;
    
    t = simulation.tvals;
    u = simulation.Fvals;
    y = simulation.disp;
    fsamp = PNLSS.Fs;
    save(['Data/SimExp_full_shaker_' Shaker '_' valorest '.mat'], 't', 'u', 'y', 'fsamp', 'PNLSS');
end
% Compare modal characteristics for experiment and Harmonic Balance methods
