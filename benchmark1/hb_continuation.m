% DESCRIPTION: 
% Investigation of a flat beam considering geometric nonlinearities.

clear; clc;
close all; 

srcpath = '../src/nlvib';
addpath(genpath(srcpath));

%% Define system

% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.01;
Nmod = 5;
setup = 'New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L_2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
fname = ['beam_New_Design_Steel_analytical_5t_' ...
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

%% Compute frequency response using harmonic balance
analysis = 'FRF';
H = 7;              % harmonic order
N=2*3*H+1;
Ntd = 1e3;

% Analysis parameters
Om_e = om(1)*.1;      % start frequency
Om_s = 5*om(1);     % end frequency

% Excitation levels
exc_lev = [30];
X = cell(size(exc_lev));
for iex=1:length(exc_lev)
    % Set excitation level
    oscillator.Fex1 = Fex1*exc_lev(iex);
    
    % Initial guess (solution of underlying linear system)
    Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
    y0 = zeros((2*H+1)*length(Q1),1);
    y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
    qscl = max(abs((-om(1)^2*M + 1i*om(1)*D + K)\oscillator.Fex1));
    
    % Solve and continue w.r.t. Om
    ds = 50; % -> better for exc_lev = 50
        
    Dscale = [1e-6*ones(length(y0),1);(Om_s+Om_e)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4);
    X{iex} = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,H,N,analysis),...
        Om_s,Om_e,ds,Sopt);
    
end

% save('frf.mat','X','ds','H','Ntd','exc_lev','oscillator','n')

%% NMA
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

save('nma.mat','Solinfo','Sol','X','ds','H','n','imod')
