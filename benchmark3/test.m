clearvars;


f1 = 50;
f2 = 405;
P  = 4;           % Periods, we need to ensure steady state

% select Nt satisfying fs as 80% of Nyquist freq. Does not work yet.
f0 = 1;
N  = ceil((f2-f1)/f0);      % freq points
Nt = ceil(1/0.5 * f2*2/f0); % Time points per cycle
fs = Nt*f0;                 % Samping frequency - implicit given

upsamp = 10;
N  = 1e2*upsamp;       % freq points
Nt = 2^12*upsamp;      % Time points per cycle
f0 = (f2-f1)/N;
fs = Nt*f0;     % Samping frequency

% set the type of multisine
ms_type = 'full';  % can be 'full', 'odd' or 'random_odd'

if fs/2/upsamp <= f2
    error('Increase sampling rate! Current f0:%g, requires atleast Nt:%g',f0,f2*2*upsamp/f0);
end


t1   = 0;
t2   = P/f0;
% time vector.
t    = linspace(t1, t2, Nt*P+1);  t(end) = [];
% t1 = t1:1/fs:t2-1/fs;
freq = (0:Nt-1)/Nt*fs;   % frequency content
nt = length(t);          % total number of points

f_NL = @(x) x + 0.2*x.^2 + 0.1*x.^3; % Nonlinear function
[b,a] = cheby1(2,10,2*0.1); % Filter coefficients

R = 2;
u = zeros(length(t),R);
for r = 1:R
[fex, ms] = multisine(f1, f2, N, 1);
for i=1:length(t)
    u(i,r) = fex(t(i));
end
end
% u = u';
x = f_NL(u); % Intermediate signal
y0 = filter(b,a,x); % Noise-free output signal
y = y0; % + 0.01*mean(std(y0))*randn(size(y0)); % Noisy output signal
y = reshape(y, [Nt,P,R,1]);
u = reshape(u, [Nt,P,R,1]);

[y,u] = dsample(y,u,upsamp);


r = 1;
[Nt,P,R,p] = size(y);
figure;
per = (y(:,1:end-1,r)-y(:,end,r)) / rms(y(:,1,r));
plot(t(1:Nt*(P-1)),db(per(:)),'k-')
% indicate periods
h1 = vline(t((1:r*(P-1))*Nt),'--g');
% indicate realizations
h2 = vline(t((1:r)*Nt*(P-1)),'--k');set([h1 h2],'LineWidth',0.5)
xlabel('time (s)')
ylabel('Relative error to last period (dB)')
title([num2str(Nt) ' samples per period'])

% plot only half spectrum(single sided spectrum)
dft = fft(u(:,1));
nt = length(dft)/2+1;
xdft = dft(1:nt);

figure; subplot(2,1,1)
plot(freq(1:nt),db(xdft),'-*')
xlabel('frequency (Hz)')
ylabel('magnitude (dB)')
title('FFT of one period of the multisine realizations')

subplot(2,1,2)
plot(freq(1:nt),angle(xdft),'-*')
xlabel('frequency (Hz)')
ylabel('phase (rad)')
title('FFT of one period of the multisine realizations')

Lines = ms.lines;
PTrans = 1;
u(:,1:PTrans,:) = []; % Remove transient period(s)
x(:,1:PTrans,:) = []; % Remove transient period(s)
y(:,1:PTrans,:) = []; % Remove transient period(s)
U = fft(u); % Input spectrum
Y = fft(y); % Output spectrum
U = U(Lines,:,:); % Select only excited frequency lines
Y = Y(Lines,:,:); % Select only excited frequency lines
U = permute(U,[1,4,3,2]); % F x m x R x P
Y = permute(Y,[1,4,3,2]); % F x p x R x P
[G,covGML,covGn] = fCovarFrf(U,Y); % Compute FRF, total and noise distortion
figure
plot(Lines,db([G(:) covGML(:) covGn(:)]))
xlabel('Frequency line')
ylabel('Amplitude (dB)')
legend('FRF','Total distortion','Noise distortion')