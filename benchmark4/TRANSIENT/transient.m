clc
clear all
% addpath('../../../../RESEARCH/ANALYSES/ROUTINES/FEM/')
% addpath('../../../../RESEARCH/ANALYSES/ROUTINES/TRANSIENT/')
addpath('../../src/nlvib/SRC/MechanicalSystems/')
addpath('../../src/transient/')

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
fs = 500;
Nfpts = (f2-f1)/df+1;
freqs = linspace(f1, f2, Nfpts);
har = ones(Nfpts, 1);
famp = 2.0;
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
             -Mb\Kb -Mb\Cb]);
         
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

%% ODE System
func = @(t, x, z) ROC_DYNSYS(t, x, z, Ab, bb, bxb, byb, bnb, fricts, fex, @(t,x,z,ff) ROC_ELDRYFRIC(t,x,z,ff));  % State velocity function

X0 = zeros(length(Ab), 1);
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
treq = linspace(0, Tmax, ceil(5*f2*Tmax)+1);
tic
% [T, X, Z] = RK_GEN_AD(func, [0, Tmax], X0, Z0, pars);
[T, X, Z] = RK_GEN_AD_TV(func, treq, X0, Z0, pars);
toc

%% Saving
Fex = fex.ffun{1}(T);
save(sprintf('./RUN%d.mat',rn), 'T', 'X', 'Z', 'Fex', 'Prds', 'f1', 'f2', 'df', ...
    'freqs', 'fex');
end

%% Resave data
fdir = 'famp20';

load(sprintf('./%s/RUN1.mat',fdir), 'f2', 'df', 'Prds', 'X');
fsamp = 5*f2;
Nt = 5*f2/df;  % Time points per period
Nd = size(X,2)/2;   % Number of dynamical DOFs
u = zeros(Nt, Prds, Repeats);
y = zeros(Nt, Prds, Repeats, Nd);
ydot = zeros(Nt, Prds, Repeats, Nd);

for rn=1:Repeats
    load(sprintf('./%s/RUN%d.mat',fdir,rn), 'T', 'X', 'Z', 'Fex', 'Prds', ...
        'f1', 'f2', 'df', 'freqs', 'fex');
    Tmax = (Prds+1)/df;
    Treq = linspace(0, Tmax, ceil(5*f2*Tmax)+1); Treq(end)=[];
    Xreq = interp1(T, X, Treq);

    u(:, :, rn) = reshape(fex.ffun{1}(Treq(Nt+1:end)), Nt, Prds);
    y(:, :, rn, :) = reshape(Xreq(Nt+1:end, 1:Nd), Nt, Prds, Nd);
    ydot(:, :, rn, :) = reshape(Xreq(Nt+1:end, Nd+(1:Nd)), Nt, Prds, Nd);
end
disp('Done!');

t = Treq(1:(Prds*Nt));
fdof = fex.dofs;
famp = fex.fval;
save(sprintf('./%s/CLCLEF_MULTISINE.mat',fdir), 'u', 'y', 'ydot', 'f1', 'f2', 'df', ...
    'fsamp', 'freqs', 't', 'famp', 'fdof');


%% Plot
fdir = 'famp20';
rn = 1;
load(sprintf('./%s/RUN%d.mat',fdir,rn), 'T', 'X', 'Z', 'Fex', 'Prds', ...
    'f1', 'f2', 'df', 'freqs', 'fex');

Tmax = (Prds+1)/df;
Treq = linspace(0, Tmax, ceil(5*f2*Tmax)+1); Treq(end)=[];
Treq = Treq(1:Nt);
Xreq = interp1(T, X, Treq);
Freq = fex.ffun{1}(Treq);

Nft = length(Freq);
Xf = fft(Xreq(1:Nt,fex.dofs));  Xf = Xf(1:(Nft/2))/(Nft/2);  Xf(1) = Xf(1)*2;
Freqf = fft(Freq);
Freqf = Freqf(1:(Nft/2))/(Nft/2);  Freqf(1) = Freqf(1)*2;
dfft = df;

figure(1)
clf()
semilogy((0:(Nft/2-1))*dfft, abs(Freqf), '-'); hold on
semilogy((0:(Nft/2-1))*dfft, abs(Xf), '-')
xlabel('Frequency (Hz)')
ylabel('Content')

legend('Forcing (N)', 'Response (m)')
% print(sprintf('%s_ts.eps',fdir), '-depsc')