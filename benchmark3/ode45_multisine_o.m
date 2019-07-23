% Simulate the system using ode5.
%
% To truly ensure periodicity(ie. steady state), it is important that you
% DO NOT use matlabs ode45 solver, as it is a variable step-size solver.
% A fixed step-size ode solver solver seeems to be working.


% close all
clear variables

srcdir = '../src/matlab';
addpath(genpath(srcdir));

dataname = 'ms_full';
savedata = true;
savefig = true;

benchmark = 2;

%% Define system
switch benchmark
    case 1
        % Excitation levels. Note 50 is too high!
        exc_lev = [1,10,20,30,40];
        % low/high freq
        f1 = 50;
        f2 = 405;
        N = 2e3;      % freq points, gives freq resolution
        Nt = 2^12;    % Time points per cycle
        upsamp = 4;   % upsampling factor to ensure integration accuracy.
    case 2
        exc_lev = [1,10,20,30];
        % natural freq (hz): ~ 264.7, 729.7, 1430.5, 2364.7, 3532.5
        % 2/4 mode not seen.
        f1 = 100;
        f2 = 405;
        % for mdof system, we need to increase time points to avoid
        % exploding sol, when doing fixed dt integration.(ie. higher fs)
        % higher fs is obtained by decreasing N or increasing Nt.
        N = 2e3;
        Nt = 2^12;
        upsamp = 8;
    case 3
        exc_lev = [1,10,20,30];
        f1 = 50;
        f2 = 405;
        N = 2e3;
        Nt = 2^12;
        upsamp = 8;
end

sys = load_sys(benchmark);
n = sys.Nmod;

%% multisine, using time domain formulation

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
    if sum(reshape(any(isnan(y)), [],1)) || ...
            sum(reshape(any(isnan(ydot)), [],1))
        fprintf('Error: simulation exploded. Try increasing Nt or upsamp\n')
        break % don't quit, we still want to save data.
    end
end
disp(['ode5 with multisine in time domain required ' num2str(toc) ' s.']);

if savedata
    save(sprintf('data/b%d_A%d_up%d_%s',benchmark,A,upsamp,dataname),...
        'u','y','ydot','f1','f2','fs','freq','t','A','sys','MS','upsamp')
end

% only plot if it's supported (ie if we're not running it from cli)
% https://stackoverflow.com/a/30240946
if usejava('jvm') && feature('ShowFigureWindows')
    % phi: convert to physical coordinate
    phi = sys.PHI([sys.L/2]);
    ms_plot(t,y,u,freq,MS{r}.lines,phi,benchmark,A,dataname,savefig)
end


%% Check the spectrum match between downsampled and original data 

% % plot center deflection
% phi = sys.PHI(sys.L/2);
% Y1 = fft(phi*Y(1:Ntint,1:n)')/(Ntint);
% Y2 = fft(phi*squeeze(y(1:Nt,1,1,:))')/(Nt);
% % n = 5;  % node to plot
% % Y1 = fft(Y(1:Ntint,n))/(Ntint);
% % Y2 = fft(y(1:Nt,1,1,n))/(Nt);
% freq = (0:Nt-1)/Nt*fs;
% nft1 = length(Y1)/2;
% nft2 = length(Y2)/2;
% freq1= (0:Ntint-1)/Ntint*fsint; 
% freq2= (0:Nt-1)/Nt*fs; 
% figure(3)
% clf
% hold on
% plot(freq1(1:nft1), db(abs(Y1(1:nft1))))
% plot(freq2(1:nft2), db(abs(Y2(1:nft2))))
% legend('Original','Downsampled')
% xlabel('Frequency (Hz)')
% ylabel('Magnitude (dB)')
% % export_fig(gcf,sprintf('fig/b%d_fft_comp_n%d',benchmark,n),'-png')
% export_fig(gcf,sprintf('fig/b%d_fft_comp',benchmark),'-png')
end
