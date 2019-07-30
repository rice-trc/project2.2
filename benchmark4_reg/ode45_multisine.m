% Simulate the system using ode5.
%
% To truly ensure periodicity(ie. steady state), it is important that you
% DO NOT use matlabs ode45 solver, as it is a variable step-size solver.
% A fixed step-size ode solver solver seeems to be working.


% close all
clear variables

srcdir = '../src/matlab';
addpath(srcdir);
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')

dataname = 'ms_full';
savedata = true;
savefig = true;

benchmark = 4;

% nat freqs for first 5 modes, Fully free system. linear. (Hz)
% 48.1 301.3 844.4 1659.2 2758.0
% nat freqs for first 5 modes, Fully impact system. (linear) (Hz)
% 62.1 322.8 847.0 1661.2 2759.8


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
T_tip = zeros(1,n); T_tip(end-2) = 1;

% Set tanh regularization parameter
oscillator.nonlinear_elements{1}.eps = 1e-4;

% Apply forcing to free end of beam in translational direction
oscillator.Fex1(end-2) = 1;
Fex1 = oscillator.Fex1;


%% multisine, using time domain formulation
exc_lev = [1,5,10,15,30];
f1 = 10;
f2 = 100;
N = 1e3;
Nt = 2^13;
upsamp = 4;

R  = 4;            % Realizations. (one for validation and one for testing)
P  = 8;           % Periods, we need to ensure steady state

Ntint = Nt*upsamp;  % Upsampled points per cycle
f0 = (f2-f1)/N;     % frequency resolution -> smaller -> better
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

q0   = zeros(n,1);
u0   = zeros(n,1);
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
for A = exc_lev
fprintf('##### A: %g\n',A);
u = zeros(Nt,P,R);
y = zeros(Nt,P,R,n);
ydot = zeros(Nt,P,R,n);

MS = cell(R, 1);

tic
for r=1:R
    fprintf('R: %d\n',r);
    % multisine force signal
    [fex, MS{r}] = multisine(f1, f2, N, A, [], [], r);
    oscillator.Fex1 = Fex1*A;
    [tout,Y] = ode45(@(t,y) odesys(t,y, fex, oscillator), tint,[q0;u0]);
    % Y = ode8(@(t,y) odesys(t,y, fex, beam), tint,[q0;u0]);
 
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
