% Simulate the system using ode5.
%
% To truly ensure periodicity(ie. steady state), it is important that you
% DO NOT use matlabs ode45 solver, as it is a variable step-size solver.
% A fixed step-size ode solver solver seeems to be working.


% close all
clear variables

srcdir = '../src/matlab';
addpath(genpath(srcdir));
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')

dataname = 'ms_full';
savedata = true;
savefig = true;

benchmark = 5;

% nat freqs for first 5 modes, Fully free system. linear. (Hz)
% 48.1 301.3 844.4 1659.2 2758.0
% nat freqs for first 5 modes, Fully impact system. (linear) (Hz)
% 62.1 322.8 847.0 1661.2 2759.8

%% Define system
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

% Nonlinearity
Nnl = 4;
dir = 'trans';
kn  = 1.3e6;
gap = 1e-3;
add_nonlinear_attachment(beam, Nnl, dir, 'unilateralspring', ...
    'stiffness', kn, 'gap', gap);
Nd = size(beam.M, 1);

%% Linearized limit cases
% Slipped
[Vsl, Dsl] = eig(beam.K, beam.M);
[Dsl, si] = sort(sqrt(diag(Dsl)));
Vsl = Vsl(:, si);  Vsl = Vsl./sqrt(diag(Vsl'*beam.M*Vsl)');

% Stuck
Knl = zeros(size(beam.M));
nli = find(beam.nonlinear_elements{1}.force_direction);
Knl(nli, nli) = kn;
Kst = beam.K + Knl;
[Vst, Dst] = eig(Kst, beam.M);
[Dst, si] = sort(sqrt(diag(Dst)));
Vst = Vst(:, si); Vst = Vst./sqrt(diag(Vst'*beam.M*Vst));

%% Rayleigh damping
% Desired
zs = [8e-3; 8e-3];
ab = [ones(length(zs),1) Dst(1:length(zs)).^2]\(2*zs.*Dst(1:length(zs)));
beam.D = ab(1)*beam.M + ab(2)*Kst;

Zetas = diag(Vst'*beam.D*Vst)./(2*Dst);
Zetas_req = 8e-3*ones(beam.n,1);
beam.D = inv(Vst')*diag(2*Dst.*Zetas_req)*inv(Vst);

n = size(beam.M,1);

%% HCB Reduction
[Mhcb, Khcb, Thcb] = HCBREDUCE(beam.M, beam.K, (Nnl-2)*2+1, 3);
beam_hcb = struct('M', Mhcb, 'K', Khcb, 'L', beam.L*Thcb, 'n', size(Khcb,1), ...
    'D', Thcb'*beam.D*Thcb, 'Fex1', Thcb'*beam.Fex1, 'nonlinear_elements', {beam.nonlinear_elements});
beam_hcb.nonlinear_elements{1}.force_direction = Thcb'*beam.nonlinear_elements{1}.force_direction;

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
% K_stinger_elem = k_stinger*[1.0 -PHI_L2;-PHI_L2' PHI_L2'*PHI_L2];
% K_stinger_elem = k_stinger*[-M\PHI_L2'*PHI_L2, M\PHI_L2';PHI_L2/M_T, -1.0/M_T];
% K_stinger_elem = k_stinger*[-M\gam'*PHI_L2, M\gam';PHI_L2/M_T, -1.0/M_T];  % Base Exitation
K_stinger_elem = k_stinger*[-beam_hcb.M\Thcb(fdof,:)'*Thcb(fdof,:), beam_hcb.M\Thcb(fdof,:)'; Thcb(fdof,:)/M_T, -1.0/M_T];

%% State-space System
smbeam.A = blkdiag([zeros(beam_hcb.n) eye(beam_hcb.n); -beam_hcb.M\beam_hcb.K, -beam_hcb.M\beam_hcb.D], A_shaker);
smbeam.A([beam_hcb.n+(1:beam_hcb.n) end], [1:beam_hcb.n end-1]) = smbeam.A([beam_hcb.n+(1:beam_hcb.n) end], [1:beam_hcb.n end-1])+K_stinger_elem;
smbeam.B = [zeros(2*beam_hcb.n,1); B_shaker(:,1)];
smbeam.C = [eye(beam_hcb.n) zeros(beam_hcb.n, size(A,1)-beam_hcb.n)];
smbeam.D = 0;
smbeam.nx = 1; smbeam.nxd = beam_hcb.n+1;
smbeam.fnls = [1;zeros(size(A,1)-1,1)];
smbeam.nonlinear_elements = beam_hcb.nonlinear_elements;
smbeam.nonlinear_elements{1}.force_direction = ...
   [zeros(beam_hcb.n,1); -beam_hcb.M\smbeam.nonlinear_elements{1}.force_direction; zeros(size(A_shaker,1),1)];
smbeam.n = size(smbeam.A,1);

%% multisine, using time domain formulation
exc_lev = [1,5,10,15,30,60,120,240];
f1 = 10;
f2 = 100;
N = 1e3;
Nt = 2^13;
upsamp = 4;

R  = 5;            % Realizations. (one for validation and one for testing)
P  = 10;           % Periods, we need to ensure steady state

Ntint = Nt*upsamp;  % Upsampled points per cycle
f0 = (f2-f1)/N;     % frequenzeros(2*beam_hcb.n,1)cy resolution -> smaller -> better
fs = Nt*f0;         % downsampled Sampling frequency
fsint = Ntint*f0;

% set the type of multisine
ms_type = 'full';  % can be 'full', 'odd' or 'random_odd'

% the Nyquist freq should hold for even for the downsampled signal
if fsint/2/upsamp <= f2
    error(['Increase sampling rate! Current f0:%g,',...
        'requires atleast Nt:%g'],f0,f2*2*upsamp/f0);
end

Pfilter = 0;
if upsamp > 1
    % one period is removed due to edge effect of downsampling
    Pfilter = 1;
end
Pint = P + Pfilter;

q0   = zeros(beam.n,1);
u0   = zeros(beam.n,1);
t1   = 0;
t2   = Pint/f0;
t = linspace(t1, P/f0, Nt*P+1);  t(end) = [];
% upsampled time vector.
tint = linspace(t1, t2, Ntint*Pint+1);  tint(end) = [];
% t1 = t1:1/fs:t2-1/fs;
freq = (0:Nt-1)/Nt*fs;   % downsampled frequency content
nt = length(t);          % total number of points

fprintf(['running ms benchmark:%d. R:%d, P:%d, Nt_int:%d, fs_int:%g, ',...
    ' f0:%g, upsamp:%d\n'],benchmark,R,P,Ntint,fsint,f0,upsamp);
for A = exc_lev(end)
fprintf('##### A: %g\n',A);
u = zeros(Nt,P,R);
sf = zeros(Nt,P,R);
y = zeros(Nt,P,R,n);
ydot = zeros(Nt,P,R,n);

MS = cell(R, 1);

tic
for r=1:R
    fprintf('R: %d\n',r);
    % multisine force signal
    [fex, MS{r}] = multisine(f1, f2, N, A, [], [], r);
%    beam.Fex1 = Fex1*A;
%     [tout,Y] = ode45(@(t,y) odesys(t,y, fex, beam), tint,[q0;u0]);
%     Y = ode8(@(t,y) odesys(t,y, fex, beam), tint,[q0;u0]);

    smbeam.B = [zeros(2*beam_hcb.n,1); B_shaker(:,1)]*A;
tic
    [tout,Y] = ode45(@(t,y) odesys_ss(t,y, fex, smbeam), tint,zeros(2*smbeam.n,1));
%    Yhcb = ode8(@(t,y) odesys_ss(t,y, fex, beam_hcb), tint,zeros(2*beam_hcb.n,1));
    Y = Yhcb*blkdiag(Thcb', Thcb');
toc
    % no need to downsample u. Just use the downsampled time vector
    u(:,:,r) = reshape(fex(t), [Nt,P]);
    if upsamp > 1
        % decimate measured signal by upsamp ratio
        ytmp = dsample(reshape(Y(:,1:n),[Ntint,Pint,1,n]),upsamp);
        y(:,:,r,:) = ytmp;
        ytmp = dsample(reshape(Y(:,n+1:end),[Ntint,Pint,1,n]),upsamp);
        ydot(:,:,r,:) = ytmp;
    else
        y(:,:,r,:) = reshape(Y(:,1:n), [Nt,P,n]);
        ydot(:,:,r,:) = reshape(Y(:,n+1:end), [Nt,P,n]);
    end
    if sum(reshape(any(isnan(y)), [],1)) || ...
            sum(reshape(any(isnan(ydot)), [],1))
        fprintf('Error: simulation exploded. Try increasing Nt or upsamp\n')
        break % don't quit, we still want to save data.
    end
end
disp(['ode5 with multisine in time domain required ' num2str(toc) ' s.']);

if savedata
    save(sprintf('data/b%d_A%d_up%d_%s',benchmark,A,upsamp,dataname),...
        'u','y','ydot','f1','f2','fs','freq','t','A','beam','MS','upsamp')
end


% only plot if it's supported (ie if we're not running it from cli)
% https://stackoverflow.com/a/30240946
if usejava('jvm') && feature('ShowFigureWindows')
    
idof = fdof;
fpath = './FIGURES/pnlss';

ms_plot(t,y,u,freq,idof,benchmark,A,dataname,savefig,fpath)

%% Check the spectrum match between downsampled and original data 
Y1 = fft(Y(1:Ntint,idof))/(Ntint);
Y2 = fft(y(1:Nt,1,1,idof))/(Nt);
freq = (0:Nt-1)/Nt*fs;
nft1 = length(Y1)/2;
nft2 = length(Y2)/2;
freq1= (0:Ntint-1)/Ntint*fsint; 
freq2= (0:Nt-1)/Nt*fs; 
figure
hold on
plot(freq1(1:nft1), db(abs(Y1(1:nft1))))
plot(freq2(1:nft2), db(abs(Y2(1:nft2))),'--')
legend('Original','Downsampled')
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
xlim([40,70])
grid minor
grid on
export_fig(gcf,sprintf('%s/b%d_A%g_fft_comp_n%d',fpath,benchmark,A,idof),'-png')
end

end
