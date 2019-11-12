% simulate the system using ode45.
%
% This requires that the odesys is written in such a way that it can handle
% adaptive time steps. This is not the case if the multisine is calculated
% a priori, fx. using PNLSS

% close all
clearvars
addpath('../src/matlab/')

%% Define system
% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.03;
Nmod = 1;
setup = './data/New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L2 = PHI(L/2);

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(1:Nmod).*om(1:Nmod));
K = diag(om.^2);

% load nonlinear coefficients (can be found e.g. analytically)
fname = ['./data/beam_New_Design_Steel_analytical_5t_' ...
    num2str(thickness*1000) 'mm.mat'];
[p, E] = nlcoeff(fname, Nmod);

% Fundamental harmonic of external forcing
Fex1 = PHI_L2';

nz = size(p,1);
n = Nmod;

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
K_stinger_elem = k_stinger*[-M\gam'*PHI_L2, M\gam';PHI_L2/M_T, -1.0/M_T];  % Base Exitation

%% State-space (continuous time) matrices
model.A = blkdiag([0 1;-K/M -D/M], A_shaker);  model.A([2 7],[1 6]) = model.A([2 7],[1 6]) + K_stinger_elem;
model.B = [0;0;B_shaker(:,1)]
model.C = [PHI_L2 0 zeros(1, 5)];
model.D = 0;
model.nlcof = struct('power', p, 'coef', E, 'fnls', [zeros(Nmod);-inv(M);zeros(5,Nmod)], 'nx', 1, 'nxd', 2);
imag(eig(model.A))

% %%
% model.A = [0 1;-K/M -D/M];
% model.B = [0; PHI_L2/M];
% model.C = [PHI_L2 0];
% model.D = 0;
% model.nlcof = struct('power', p, 'coef', E);

%% multisine, using time domain formulation

upsamp = 1;  % Upsampling factor
R  = 4;           % Realizations. (one for validation and one for testing)
P  = 8;           % Periods, we need to ensure steady state
f1 = 200;          % low freq
f2 = 700;        % high freq
fs = 1200;       % 5*f2. Must be fs>2*f2. Nyquist freq, you know:)
N  = 1e3;         % freq points
f0 = (f2-f1)/N;
A = 0.01;  % Signal RMS amplitude
Alevels = [0.01 0.25 0.50 0.75]*1000;  % 0.75: not periodic
for A=Alevels(1:end)
    fprintf('A = %f\n', A);
    Nt = 2^15;      % Time per cycle  (2^13 for 4096; 2^15 for 16384)
    fs = Nt*f0;     % Samping frequency

    Ntint = Nt*upsamp;
    fsint = Ntint*f0;

    if fs/2 <= f2
        error('Increase sampling rate!');
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
    t2   = P/f0;
    t    = linspace(t1, t2, Nt*P+1);  t(end) = [];   % time vector. ode45 interpolates output
    % Upsampled time vector
    tint = linspace(t1, t2, Ntint*Pint+1); tint(end) = [];
    freq = (0:Nt/2)*f0;   % frequency content
    nt   = Nt*P;

    sf   = zeros(Nt,P,R);
    u    = zeros(Nt,P,R);
    y    = zeros(Nt,P,R,n);
    ydot = zeros(Nt,P,R,n);
    tic
    MS = cell(R, 1);
    for r=1:R
        % multisine force signal
        [fex, MS{r}] = multisine(f1, f2, N, A, [], [], r);

        model.fex = fex;
    %     [tout,Y] = ode45(@(t,y) sys(t,y, par), t,[q0;u0]);
    tic
        [tout,Y] = ode45(@(t,y) sys_ss(t,y, model), tint,[q0;u0;zeros(5,1)]);
%         Y = ode5(@(t,y) sys_ss(t,y, model), tint, [q0;u0;zeros(5,1)]);
    toc

        u(:,:,r) = reshape(fex(t'), [Nt,P]);
        sf(:,:,r) = reshape(gam*k_stinger*(Y(:,6)-PHI_L2*Y(:,1)), [Nt, P]);
        if upsamp > 1
            ytmp = dsample(reshape(Y(:,1:n), [Ntint,Pint,1,n]), upsamp);
            y(:,:,r,:) = ytmp;
            ytmp = dsample(reshape(Y(:,n+1:(2*n)), [Ntint,Pint,1,n]), upsamp);
            ydot(:,:,r,:) = ytmp;
        else
            y(:,:,r,:) = reshape(Y(:,1:n), [Nt,P,n]);
            ydot(:,:,r,:) = reshape(Y(:,n+1:(2*n)), [Nt,P,n]);
        end
    end
    disp(['ode45 with multisine in time domain required ' num2str(toc) ' s.']);

    save(sprintf('data/ode45_multisine_shaker_A%.2f_F%d.mat',A,fs),'u','y','ydot','sf','f1','f2','fs','freq',...
        't','A','PHI_L2', 'MS', 'model', 'Nt','f0','Y')

    %% show time series
    sprintf('data/ode45_multisine_A%.2f_F%d.mat',A,fs)
    load(sprintf('data/ode45_multisine_shaker_A%.2f_F%d.mat',A,fs))
    r = 1;
    Y = PHI_L2*reshape(y(:,:,r,:),[],n)';

    figure;
    plot(t, Y,'k-')
    % indicate periods
    h1 = vline(t((1:r*P)*Nt),'--g');
    % indicate realizations
    h2 = vline(t((1:r)*Nt*P),'--k');set([h1 h2],'LineWidth',0.5)
    xlabel('time (s)')
    ylabel('magnitude')
    title(['Multisine: ' num2str(r) ' realizations of ' num2str(P) ' periods of ' num2str(Nt) ' samples per period'])
    % export_fig('fig/multisine_sim_time.pdf')

    % show periodicity
    Y1 = reshape(Y,[Nt,P,r]);
    figure;
    per = (Y1(:,1:end-1,1)-Y1(:,end,1)) / rms(Y1(:,1,1));
    plot(t(1:Nt*(P-1)),db(per(:)),'k-')
    % indicate periods
    h1 = vline(t((1:r*(P-1))*Nt),'--g');
    % indicate realizations
    h2 = vline(t((1:r)*Nt*(P-1)),'--k');set([h1 h2],'LineWidth',0.5)
    xlabel('time (s)')
    ylabel('Relative error to last period (dB)')
    title([num2str(Nt) ' samples per period'])
    % export_fig('fig/multisine_periodicity_lowfreq.pdf')

    % force signal for one period
    % plot only half spectrum(single sided spectrum)
    dft = fft(u(:,1,r));
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
    % export_fig('fig/multisine_freq.pdf')
end

