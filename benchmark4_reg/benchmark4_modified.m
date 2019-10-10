%========================================================================
% DESCRIPTION:
% Friction Damped System
%
% For More Information on NLvib see
% https://www.ila.uni-stuttgart.de/nlvib
%
% Developer:    Malte Krack
% Copyright:    Institute of Aircraft Propulsion Systems
%               University of Stuttgart, Germany
%========================================================================

clearvars; close all; clc; addpath('SRC'); addpath('SRC/MechanicalSystems');
set(0,'DefaultLineLineWidth',2)

%% Define system

% Properties of the beam
len = 2;                % length
height = .05*len;       % height in the bending direction
thickness = 3*height;   % thickness in the third dimension
E = 185e9;              % Young's modulus
rho = 7830;             % density
BCs = 'clamped-free';   % constraints

% Setup one-dimensional finite element model of an Euler-Bernoulli beam
n_nodes = 9;            % number of equidistant nodes along length
beam = FE_EulerBernoulliBeam(len,height,thickness,E,rho,...
    BCs,n_nodes);

% Attach additional mass and spring and apply dry friction element
kstatic = 3*E*thickness*height^3/12/len^3;
icpl = length(beam.M)-1;
m = .02*beam.M(icpl,icpl); k = kstatic*2e-1;
M = blkdiag(beam.M,m);
K = blkdiag(beam.K,0);
K([icpl end],[icpl end]) = K([icpl end],[icpl end]) + [1 -1;-1 1]*k;
n = length(M);
w = zeros(n,1); w(end) = 1;
muN = 1e2;
nonlinear_elements = struct('type','tanhDryFriction',...
    'friction_limit_force',muN,'force_direction',w);
oscillator = MechanicalSystem(M,0*M,K,nonlinear_elements,...
    zeros(length(M),1));

% Vectors recovering deflection at tip and nonlinearity location
T_nl = oscillator.nonlinear_elements{1}.force_direction';
% T_tip = zeros(1,n); T_tip(end-2) = 1;
%% Modal analysis the linearized system

% Modes for free sliding contact
[PHI_free,OM2] = eig(oscillator.K,oscillator.M);
om_free = sqrt(diag(OM2));
% Sorting
[om_free,ind] = sort(om_free); PHI_free = PHI_free(:,ind);

% Modes for fixed contact
inl = find(T_nl); B = eye(length(oscillator.M)); B(:,inl) = [];
[PHI_fixed,OM2] = eig(B'*oscillator.K*B,B'*oscillator.M*B);
om_fixed = sqrt(diag(OM2));
% Sorting
[om_fixed,ind] = sort(om_fixed); PHI_fixed = B*PHI_fixed(:,ind);

%% Nonlinear modal analysis using harmonic balance

analysis = 'NMA';

% Analysis parameters
H = 7;              % harmonic order
Ntd = 2^12;         % number of time samples per period
imod = 1;           % mode to be analyzed
log10a_s = -3;      % start vibration level (log10 of modal mass)
log10a_e = -1;      % end vibration level (log10 of modal mass)
inorm = 2;          % coordinate for phase normalization

% Initial guess vector x0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
n = length(oscillator.M);
om = om_fixed(imod); phi = PHI_fixed(:,imod);
Psi = zeros((2*H+1)*n,1);
Psi(n+(1:n)) = phi;
x0 = [Psi;om;0];

% Set tanh regularization parameter
oscillator.nonlinear_elements{1}.eps = 1e-4;

% Solve and continue w.r.t. Om
ds = .01;
fscl = 1/mean(abs(oscillator.K*phi));
X_HB = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,H,Ntd,analysis,inorm,fscl),...
    log10a_s,log10a_e,ds,struct('dynamicDscale',1,'stepmax',2e2));

% Interpret solver output
Psi_HB = X_HB(1:end-3,:);
om_HB = X_HB(end-2,:);
del_HB = X_HB(end-1,:);
log10a_HB = X_HB(end,:);
a_HB = 10.^log10a_HB;
Q_HB = Psi_HB.*repmat(a_HB,size(Psi_HB,1),1);

save('data/nma_fricbeam');

%% Nonlinear frequency response analysis using Harmonic Balance
T_tip = zeros(1,n); T_tip(end-2) = 1;
analysis = 'FRF';

% Analysis parameters
H = 7;                     % harmonic order
N = 2^12;                    % number of time samples per period
Om_s = 1.1*om_fixed(1);     % start frequency
Om_e = .9*om_free(1);       % end frequency

levels = logspace(1,log10(3e2),6);
OM_HB = cell(size(levels));
Qtip_rms_HB = OM_HB; Q_HB = OM_HB;

for ii=1:length(levels)
% Apply forcing to free end of beam in translational direction, with
% magnitude 'fex'
oscillator.Fex1(end-2) = levels(ii);

% Specify stiffness proportional damping corresponding to D=1% at linear 
% at the first linearized resonance
beta   = 2*1e-2/om_fixed(1);
oscillator.D = beta*oscillator.K;

% Initial guess (from underlying linear system)
Fex1 = oscillator.Fex1;
Q1 = B*((B'*(-Om_s^2*oscillator.M + 1i*Om_s*oscillator.D + oscillator.K)*B)\(B'*Fex1));
Q1_free = (-Om_s^2*oscillator.M + 1i*Om_s*oscillator.D + oscillator.K)\Fex1;
qscl = mean(abs(Q1));
x0 = zeros((2*H+1)*size(Q1,1),1);
x0(size(Q1,1)+(1:2*size(Q1,1))) = [real(Q1);-imag(Q1)];
qref = [real(Q1);-imag(Q1)];

% Solve and continue w.r.t. Om
ds = .5;

% Set options of path continuation
Sopt = struct('flag',0,'stepadapt',0,...
    'Dscale',[1e0*qscl*ones(size(x0));Om_s],...
    'dynamicDscale',1);
X_HB = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,H,N,analysis),...
    Om_s,Om_e,ds,Sopt);

% Interpret solver output
OM_HB{ii} = X_HB(end,:);
Q_HB{ii} = X_HB(1:end-1,:);

% Determine harmonics and root-mean-square value of tip displacement
Qtip_HB = kron(eye(2*H+1),T_tip)*Q_HB{ii};
Qtip_rms_HB{ii} = sqrt(sum(Qtip_HB.^2))/sqrt(2);
end

%% NMA

sdat = load('data/nma_fricbeam');
om_NM1 = sdat.om_HB;
Qtip_NM1 = kron(eye(2*sdat.H+1),T_tip)*sdat.Q_HB;
Qtip_rms_NM1 = sqrt(sum(Qtip_NM1.^2))/sqrt(2);

%% Nonlinear modal synthesis

% Determine discrete nonlinear modal data
% natural frequency
om_d = om_NM1;
% moddal damping ratio
D_d = sdat.del_HB;
% mass normalized vector psi1 and associated modal amplitude a
psi1_d = sdat.Q_HB(n+(1:n),:) - 1i*sdat.Q_HB(2*n+(1:n),:);
a_d = real(sqrt(sum(conj(psi1_d).*(M*psi1_d),1)));
psi1_d = psi1_d./repmat(a_d,size(psi1_d,1),1);
% associated tip response level
Qtip_d = kron(eye(2*sdat.H+1),T_tip)*sdat.Q_HB;
Qtip_rms_d = sqrt(sum(Qtip_d.^2))/sqrt(2);

a = a_d; om_a = om_d; D_a = D_d; psi1_a = psi1_d; Qtip_rms_a = Qtip_rms_d;
dD_a = real(sum(conj(psi1_a).*(oscillator.D*psi1_a)));

%% Determine frequency response
p_2 = om_a.^2 - (2*D_a.*om_a+dD_a).^2/2;
om4 = om_a.^4;
gam_ex = oscillator.Fex1; gam_ex(end-2) = 1;
ex = abs(gam_ex(:).'*conj(psi1_a)).^2./a.^2;
Om_NMS = cell(size(levels)); Qtip_rms_NMS = Om_NMS;

%% Illustration

figure; hold on
plot(om_NM1/om_fixed(imod),log10(Qtip_rms_NM1),'b-');
for ii=1:length(levels)
    % Set current excitation level
    Fex1_mod_a = ex*levels(ii).^2;
    
    % Solve explicitly for Om12
    Om1 = sqrt( p_2 - sqrt(p_2.^2-om4+Fex1_mod_a) );
    Om2 = sqrt( p_2 + sqrt(p_2.^2-om4+Fex1_mod_a) );
    
    % Limit to valid results
    Om = [Om1(imag(Om1)==0) Om2(imag(Om2)==0)];
    resp = [Qtip_rms_a(imag(Om1)==0) Qtip_rms_a(imag(Om2)==0)];
    [Om,ind] = sort(Om); resp = resp(ind);
    Om_NMS{ii} = Om;
    Qtip_rms_NMS{ii} = resp;
    
    plot(OM_HB{ii}/om_fixed(imod),log10(Qtip_rms_HB{ii}),'k-');%'-','color',col(ii,:));
    plot(Om_NMS{ii}/om_fixed(imod),log10(Qtip_rms_NMS{ii}),'g.','markersize',8);%'x','color',col(ii,:));
end
xlabel('\Omega/\omega'); ylabel('log_{10} a^{rms}_t');
set(gca,'ylim',[min(log10(Qtip_rms_NM1)) max(log10(Qtip_rms_NM1))],...
    'xlim',sort([Om_s Om_e])/om_fixed(imod));
legend('backbone','reference','NM-ROM');
