function [oscillator, modal] = cantilever_beam(savesys)

try, savesys; catch, savesys = false; end

% Properties of the beam
len = 2;                % length
height = .05*len;       % height in the bending direction
thickness = 3*height;   % thickness in the third dimension
E = 185e9;              % Young's modulus
rho = 7830;             % density
BCs = 'clamped-free';   % constraints

% Setup one-dimensional finite element model of an Euler-Bernoulli beam
n_nodes = 2;            % number of equidistant nodes along length
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
eps_reg = 1e-4;  % % Set tanh regularization parameter
nonlinear_elements = struct('type','tanhDryFriction',...
    'friction_limit_force',muN,'force_direction',w,'eps',eps_reg);

% Vectors recovering deflection at tip and nonlinearity location
% T_nl = oscillator.nonlinear_elements{1}.force_direction';
T_nl = w';
T_tip = zeros(1,n); T_tip(end-2) = 1;

%% Modal analysis the linearized system
% Modes for free sliding contact
[PHI_free,OM2] = eig(K,M);
om_free = sqrt(diag(OM2));
% Sorting
[om_free,ind] = sort(om_free); PHI_free = PHI_free(:,ind);

% Modes for fixed contact
inl = find(T_nl); B = eye(length(M)); B(:,inl) = [];
[PHI_fixed,OM2] = eig(B'*K*B,B'*M*B);
om_fixed = sqrt(diag(OM2));
% Sorting
[om_fixed,ind] = sort(om_fixed); PHI_fixed = B*PHI_fixed(:,ind);

%% define system
% Specify stiffness proportional damping corresponding to D=1% at linear 
% at the first linearized resonance
beta   = 2*1e-2/om_fixed(1);
C = beta*K;

oscillator = MechanicalSystem(M,C,K,nonlinear_elements,...
    zeros(length(M),1));

% Apply forcing to free end of beam in translational direction
oscillator.Fex1(end-2) = 1;

modal.om_free = om_free;
modal.PHI_free = PHI_free;
modal.om_fixed = om_fixed;
modal.PHI_fixed = PHI_fixed;
if savesys
    Fex1 = oscillator.Fex1;
    save('data/system.mat','M','C','K','w','muN','eps_reg','T_tip',...
         'Fex1','om_free','om_fixed')
end

end