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
%% multisine, using time domain formulation
exc_lev = [1,5,10,15,30];
f1 = 10;
f2 = 100;
N = 1e3;
Nt = 2^13;
upsamp = 4;

R  = 5;            % Realizations. (one for validation and one for testing)
P  = 10;           % Periods, we need to ensure steady state

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
    beam.Fex1 = Fex1*A;
    [tout,Y] = ode45(@(t,y) odesys(t,y, fex, beam), tint,[q0;u0]);
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
