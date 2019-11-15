clc
clear all
% addpath('../../../../RESEARCH/ANALYSES/ROUTINES/FEM/')
% addpath('../../../../RESEARCH/ANALYSES/ROUTINES/TRANSIENT/')
addpath('../../src/nlvib/SRC/MechanicalSystems/')
addpath('../../src/transient/')
addpath('../../src/matlab/')

%% System Properties
len = 0.70;
hgt = 0.03;
thk = hgt;
E   = 185e9;
rho = 7830.0;

Ar = thk*hgt;
I  = hgt^3*thk/12;

Nn  = 8;
Ne  = Nn-1;

% Nonlinearity
Nnl = 4;
kt = 1.3e6;
kn = 1.3e6; % Irrelevant here
mu = 1.0;
N0 = 1.0;

% MultiSine Excitation
Nex = 8;
f1 = 20;
f2 = 100;
df = 0.1;
% fs = 4096;
fs = 500;
Nfpts = (f2-f1)/df+1;
freqs = linspace(f1, f2, Nfpts);
har = ones(Nfpts, 1);
famp = 0.01*5;
%% Finite Element Model
Ndof = Nn*3;
Xcs  = linspace(0, len, Nn);  % X Coordinates

% Model Setup
M = sparse(Ndof, Ndof);
K = sparse(Ndof, Ndof);
Me = sparse(6, 6);
Ke = sparse(6, 6);
for e=1:Ne
    [Me, Ke] = EBBEAM_MATS(rho, E, Ar, I, len/Ne);
    
    M((e-1)*3 + (1:6), (e-1)*3 + (1:6)) = M((e-1)*3 + (1:6), (e-1)*3 + (1:6)) + Me;
    K((e-1)*3 + (1:6), (e-1)*3 + (1:6)) = K((e-1)*3 + (1:6), (e-1)*3 + (1:6)) + Ke;
end
% Boundary Conditions
Bc = speye(Ndof);
Bc(:, 1:3) = [];
Mb = Bc'*M*Bc;
Kb = Bc'*K*Bc;
%% Linearized limit Cases
% Slipped
[Vsl, Dsl] = eig(full(Kb), full(Mb));
[Dsl, si] = sort(sqrt(diag(Dsl)));
Vsl = Vsl(:, si);  Vsl = Vsl./sqrt(diag(Vsl'*Mb*Vsl)');

% Stuck
Knl = zeros(size(M));
Knl((Nnl-1)*3+1, (Nnl-1)*3+1) = kn;
Knl((Nnl-1)*3+2, (Nnl-1)*3+2) = kt;
Kst = Kb + Bc'*Knl*Bc;
[Vst, Dst] = eig(full(Kst), full(Mb));
[Dst, si] = sort(sqrt(diag(Dst)));
Vst = Vst(:, si); Vst = Vst./sqrt(diag(Vst'*Mb*Vst));

%% Rayleigh damping
% Desired
zs = [8e-3; 2e-3];
ab = [ones(length(zs),1) Dst(1:length(zs)).^2]\(2*zs.*Dst(1:length(zs)));
Cb = ab(1)*Mb + ab(2)*Kst;
Zetas = diag(Vst'*Cb*Vst)./(2*Dst);

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

% Stinger Element Connection
K_stinger_elem = k_stinger*[-1 1;1.0/M_T -1.0/M_T];

%% Friction Model Parameters
% SINGLE FRICTIONAL ELEMENT AT THE END 
fricts.txdofs = [(Nnl-1)*3+2]-3;
fricts.tydofs = [1];  
fricts.ndofs  = [(Nnl-1)*3+1]-3;
fricts.txwgts = 1.0; 
fricts.tywgts = 0.0; % no y tangential dof
fricts.nwgts  = 1.0;
fricts.nst    = 1;
fricts.sxdof  = 1;
fricts.sydof  = 0; % no y in state 
fricts.sndof  = 0; % no n in state
fricts.ktx    = [kt];
fricts.kty    = [kt];
fricts.kn     = [kn];
fricts.cn     = 0;
fricts.mu     = [mu];
fricts.N0     = [N0];
fricts.nel    = 1;

%% Multisine Excitation
Repeats = 4;
for rn = 1:Repeats
rng(rn);
fex.dofs  = [(Nex-1)*3+2]-3;
fex.fval  = famp;
fex.fpars = [f1 f2];
fex.nf    = 1;

phase    = 2*pi*rand(Nfpts,1);
fex.ffun = {@(t) har'*fex.fval*cos(2*pi*freqs'*t + phase) / sqrt(sum(har.^2/2))};

%% State-Space Model
Ab = sparse([zeros(size(Mb)) eye(size(Mb));
             -Kb -Cb]);  % need to premultiply with Mb
         
% force to state velocity forcing
bb = zeros(size(Ab, 1), fex.nf);
bb(size(Mb,1)+fex.dofs, 1) = eye(fex.nf);
bb((size(Mb,1)+1):end, :) = Mb\bb((size(Mb,1)+1):end, :);

% transformation matrices for nonlinear forcing
bnb = zeros(size(Ab,1), fricts.nel);
bxb = zeros(size(Ab,1), fricts.nel);
byb = zeros(size(Ab,1), fricts.nel);
bnb(size(Mb,1)+fricts.ndofs,:)  = eye(fricts.nel);
bxb(size(Mb,1)+fricts.txdofs,:) = eye(fricts.nel);
bnb((size(Mb,1)+1):end,:) = -Mb\bnb((size(Mb,1)+1):end,:);
bxb((size(Mb,1)+1):end,:) = -Mb\bxb((size(Mb,1)+1):end,:);

%% Adding Shaker
As = blkdiag(Ab, A_shaker);
As([size(Mb,1)+fex.dofs end], [fex.dofs end-1]) = As([size(Mb,1)+fex.dofs end], [fex.dofs end-1]) + K_stinger_elem;
As(size(Mb,1)+(1:size(Mb,1)),:) = Mb\As(size(Mb,1)+(1:size(Mb,1)),:);

bs = [bb*0; B_shaker(:,1)];
bns = [bnb; B_shaker(:,1)*0];
bxs = [bxb; B_shaker(:,1)*0];
bys = [byb; B_shaker(:,1)*0];
%% ODE System
func = @(t, x, z) ROC_DYNSYS(t, x, z, As, bs, bxs, bys, bns, fricts, fex, @(t,x,z,ff) ROC_ELDRYFRIC(t,x,z,ff));  % State velocity function

X0 = zeros(length(As), 1);
Z0 = zeros(fricts.nst, 1);

%% Time integrator (RK4(5) Fehlberg)
% Butcher Tableau
pars.a = [0 0 0 0 0 0; 
          1/4 0 0 0 0 0; 
          3/32 9/32 0 0 0 0; 
          1932/2197 -7200/2197 7296/2197 0 0 0;
          439/216 -8 3680/513 -845/4104 0 0;
         -8/27 2 -3544/2565 1859/4104 -11/40 0];
pars.b = [16/135 0 6656/12825 28561/56430 -9/50 2/55];
pars.bs = [25/216 0 1408/2565 2197/4104 -1/5 0];
pars.c = [0 1/4 3/8 12/13 1 1/2];
% Step size controls
pars.abstol = 1e-6;
pars.pow = 1/4;
pars.maxstep = 1e-3;
% Display
pars.Display = 'min';

% Max Simulation time
Prds = 8;
Tmax = (Prds+1)/df;
% treq = linspace(0, Tmax, ceil(5*f2*Tmax)+1);
treq = linspace(0, Tmax, ceil(fs*Tmax)+1);
tic
% [T, X, Z] = RK_GEN_AD(func, [0, Tmax], X0, Z0, pars);
[T, X, Z] = RK_GEN_AD_TV(func, treq, X0, Z0, pars);
toc

%% Saving
Fex = fex.ffun{1}(T);
save(sprintf('./RUN%d_shaker.mat',rn), 'T', 'X', 'Z', 'Fex', 'Prds', 'f1', 'f2', 'df', ...
    'freqs', 'fex', 'fs');
end
return
%% Resave data
fdir = 'famp2000_s';
load(sprintf('./%s/RUN1_shaker.mat',fdir), 'f2', 'df', 'Prds', 'X','fs');
fsamp = fs;
Nt = fsamp/df;  % Time points per period
Nd = (size(X,2)-5)/2;   % Number of dynamical DOFs
sf = zeros(Nt, Prds, Repeats);
u = zeros(Nt, Prds, Repeats);
y = zeros(Nt, Prds, Repeats, Nd);
ydot = zeros(Nt, Prds, Repeats, Nd);

for rn=1:Repeats
    load(sprintf('./%s/RUN%d_shaker.mat',fdir,rn), 'T', 'X', 'Z', 'Fex', 'Prds', ...
        'f1', 'f2', 'df', 'freqs', 'fex');
    Tmax = (Prds+1)/df;
    Treq = linspace(0, Tmax, ceil(fsamp*Tmax)+1); Treq(end)=[];
    Xreq = interp1(T, X, Treq);

    u(:, :, rn) = reshape(fex.ffun{1}(Treq(Nt+1:end)), Nt, Prds);
    sf(:, :, rn) = reshape(k_stinger*(Xreq(Nt+1:end, end-1)-Xreq(Nt+1:end,fex.dofs)), Nt, Prds, 1);
    y(:, :, rn, :) = reshape(Xreq(Nt+1:end, 1:Nd), Nt, Prds, Nd);
    ydot(:, :, rn, :) = reshape(Xreq(Nt+1:end, Nd+(1:Nd)), Nt, Prds, Nd);
end

t = Treq(1:(Prds*Nt));
fdof = fex.dofs;
famp = fex.fval;
save(sprintf('./%s/CLCLEF_SHAKER_MULTISINE.mat',fdir), 'u', 'sf', 'y', 'ydot', 'f1', 'f2', 'df', ...
    'fsamp', 'freqs', 't', 'famp', 'fdof');
disp('Done!');
%% Plot
fdir = 'famp001_n';
rn = 1;
load(sprintf('./%s/RUN%d.mat',fdir,rn), 'T', 'X', 'Z', 'Fex', 'Prds', ...
    'f1', 'f2', 'df', 'freqs', 'fex','fs');

Nt = fs/df;  % Time points per period
Tmax = (Prds+1)/df;
Treq = linspace(0, Tmax, ceil(fs*Tmax)+1); Treq(end)=[];
Xreq = interp1(T, X, Treq);
Freq = fex.ffun{1}(Treq(1:Nt));

Nft = length(Freq);
Xf = fft(Xreq(1:Nt,fex.dofs));  Xf = Xf(1:(Nft/2))/(Nft/2);  Xf(1) = Xf(1)*2;
Freqf = fft(Freq);
Freqf = Freqf(1:(Nft/2))/(Nft/2);  Freqf(1) = Freqf(1)*2;
dfft = df;

figure(1)
clf()
semilogy((0:(Nft/2-1))*dfft, abs(Freqf)/fex.fval, 'k-'); hold on
semilogy((0:(Nft/2-1))*dfft, abs(Xf)/fex.fval, '.'); hold on
xlabel('Frequency (Hz)')
ylabel('Content')

xlim([0 f2*4]);
% ylim([1e-7 1e-4])

figure(2)
clf()
plot(Treq, Xreq(:,fex.dofs), 'k.-')
h1 = vline(Treq((1:Prds)*Nt), '--g')
h2 = vline(Treq((1:1)*Nt*Prds), '--g'); set([h1 h2], 'LineWidth',0.5)

figure(3)
clf()
Yp = reshape(Xreq(1:Nt*Prds,fex.dofs), [Nt, Prds]);
per = (Yp(:,1:end)-Yp(:,end))/rms(Yp(:,1));
plot(Treq(1:Nt*Prds), db(per(:)), 'k-');
h1 = vline(Treq((1:Prds)*Nt), '--g')
h2 = vline(Treq((1:1)*Nt*Prds), '--g'); set([h1 h2], 'LineWidth',0.5)
legend('Forcing (N)', 'Response (m)')
% print(sprintf('%s_ts.eps',fdir), '-depsc')
