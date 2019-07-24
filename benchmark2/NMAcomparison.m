clc
clear all

addpath('../src/matlab/')
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

%% Define system

% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.01;
Nmod = 5;
setup = './data/New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L_2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
fname = ['./data/beam_New_Design_Steel_analytical_5t_' ...
    num2str(thickness*1000) 'mm.mat'];
[p, E] = nlcoeff(fname, Nmod);

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(:).*om(:));
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = gam;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);

% Number of degrees of freedom
n = oscillator.n;

%% Shaker Model
shaker.L   = 140e-6;      % Inductivity (H)
shaker.R   = 3.00;        % Resistance
shaker.G   = 15.48;       % Force constant (N/A)
shaker.M_T = 0.0243;      % Table Mass [kg]
shaker.M_C = 0.0190;      % Coil Mass [kg]
shaker.K_C = 8.4222*10^7; % spring constant Coil-Table [N/m]
shaker.K_T = 20707;       % spring constant Table-Ground [N/m]
shaker.D_C = 57.1692;     % damping constant Coil-Table [Ns/m]
shaker.D_T = 28.3258;     % damping constant Table-Ground [Ns/m]

% matrices (without stinger influence
M_shaker = [0 0 0;...
            0 shaker.M_C 0;...
            0 0 shaker.M_T];
C_shaker = [shaker.L 0 0;...
            0 shaker.D_C -shaker.D_C;...
            0 -shaker.D_C shaker.D_C+shaker.D_T];
K_shaker = [shaker.R shaker.G 0;...
            -shaker.G shaker.K_C -shaker.K_C;...
            0 -shaker.K_C shaker.K_C+shaker.K_T];
% C_shaker = [shaker.L shaker.G 0;...
%             0 shaker.D_C -shaker.D_C;...
%             0 -shaker.D_C shaker.D_C+shaker.D_T];
% K_shaker = [shaker.R 0 0;...
%             -shaker.G shaker.K_C -shaker.K_C;...
%             0 -shaker.K_C shaker.K_C+shaker.K_T];


% stinger parameters
E_stinger = 210000 ;%in N/mm^2
A_stinger = pi*2^2; % in mm
l_stinger = 0.0200; %in m
k_stinger = (E_stinger*A_stinger)/l_stinger;

% oscillator definition
M_full = blkdiag(M_shaker, M);
C_full = blkdiag(C_shaker, D);
K_full = blkdiag(K_shaker, K);

K_full(3:end, 3:end) = K_full(3:end,3:end) + ...
    k_stinger*[1 -PHI_L_2;...
               -PHI_L_2' PHI_L_2'*PHI_L_2];
p_full = [ones(size(p,1),3) p];
E_full = [zeros(size(E,1),3) E];
Fex1_full = [1; zeros(Nmod+2,1)];
shaker_oscillator = System_with_PolynomialStiffnessNonlinearity(M_full, C_full, K_full, p_full, E_full, Fex1_full);

n_full = shaker_oscillator.n;

%% NMA of Exact Model
H=7;
N=2*3*H+1;

% Linear modal analysis
[PHI_lin,OM2] = eig(oscillator.K,oscillator.M);
[om_lin,ind] = sort(sqrt(diag(OM2)));
PHI_lin = PHI_lin(:,ind);

analysis = 'NMA';

imod = 1;           % mode to be analyzed
log10a_s = -7;    % start vibration level (log10 of modal mass)
log10a_e = -3.2;       % end vibration level (log10 of modal mass)
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
[X,Solinfo,Sol] = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,H,N,analysis,inorm),...
    log10a_s,log10a_e,ds, Sopt);

%% NMA of Inexact (full) Model
H=7;
N=2*3*H+1;

% Linear modal analysis
[PHI_lin_f,OM2_f] = eig(shaker_oscillator.K,shaker_oscillator.M);
[om_lin_f,ind_f] = sort(sqrt(diag(OM2_f)));
PHI_lin_f = PHI_lin_f(:,ind);

analysis = 'NMA';

imod = 1;           % mode to be analyzed
log10a_s = -5;    % start vibration level (log10 of modal mass)
log10a_e = -2.5;       % end vibration level (log10 of modal mass)
inorm = 1;          % coordinate for phase normalization

% Initial guess vector y0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
om = om_lin_f(imod); phi = PHI_lin_f(:,imod);
% phi = [zeros(3,1); PHI_lin(:,imod)];
Psi = zeros((2*H+1)*n_full,1);
Psi(n_full+(1:n_full)) = phi;
x0 = [Psi;om;0];

ds      = 1.;
Sopt    = struct('Dscale',[1e-6*ones(size(x0,1)-2,1);1;1e-1;1],...
    'dynamicDscale',1,'stepmax',5e4);
[X_f,Solinfo,Sol_f] = solve_and_continue(x0,...
    @(X) HB_residual(X,shaker_oscillator,H,N,analysis,inorm),...
    log10a_s,log10a_e,ds, Sopt);