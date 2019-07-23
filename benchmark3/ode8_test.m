% Simulate the system using ode5.
%
% To truly ensure periodicity(ie. steady state), it is important that you
% DO NOT use matlabs ode45 solver, as it is a variable step-size solver.
% A fixed step-size ode solver solver seeems to be working.


% close all
clear variables

srcdir = '../src/matlab';
addpath(genpath(srcdir));

dataname = 'ode8_test';
savedata = true;
savefig = true;

benchmark = 1;
exc_lev = [1,10,20,30,40];
% low/high freq
f1 = 50;
f2 = 405;

sys = load_sys(benchmark);
n = sys.Nmod;

%% multisine, using time domain formulation

R  = 4;           % Realizations. (one for validation and one for testing)
P  = 8;           % Periods, we need to ensure steady state

% upsampling factor to ensure integration accuracy.
upsamp = 1;
N  = 2e3;           % freq points
Nt = 2^13;          % Time points per cycle
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
    
u = zeros(Nt,P,R);
y = zeros(Nt,P,R,n);
ydot = zeros(Nt,P,R,n);

MS = cell(R, 1);

tic
for r=1:R
    % multisine force signal
    [fex, MS{r}] = multisine(f1, f2, N, A, [], [], r);

    par = struct('M',sys.M,'C',sys.D,'K',sys.K,'p',sys.p,'E',sys.E,'fex',fex, 'amp', sys.Fex1);
%     [tout,Y] = ode45(@(t,y) odesys(t,y, par), t,[q0;u0]);
    Y = ode8(@(t,y) odesys(t,y, par), tint,[q0;u0]);
 
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
end
disp(['ode8 with multisine in time domain required ' num2str(toc) ' s.']);

if savedata
    save(sprintf('data/b%d_A%d_%s',benchmark,A,dataname),'u','y','ydot',...
        'f1','f2','fs','freq','t','A','sys','MS','upsamp')
end

% only plot if it's supported (ie if we're not running it from cli)
% https://stackoverflow.com/a/30240946
if usejava('jvm') && feature('ShowFigureWindows')
    % phi: convert to physical coordinate
    phi = sys.PHI([sys.L/2]);
    ms_plot(t,y,u,freq,MS{r}.lines,phi,benchmark,A,dataname,savefig)
end

end
