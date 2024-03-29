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
Dmod = [.38 .09 .08 .08]*.01;
Nmod = 3;
setup = './data/New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,~] = beams_for_everyone(setup,Nmod*2-1,thickness);
PHI_L2 = PHI(L/2);
PHI_L2 = PHI_L2(1:2:end);
om = om(1:2:end);

thickness = .001;
R=3;

% load nonlinear coefficients (can be found e.g. via IC-method)
modelname  = ['data/','beam_msh_80_4_1_3t_steel_',...
    num2str(thickness*1000),'mm_R',num2str(R),'m.mat'];
[p, E] = nlcoeff(modelname, Nmod);

% om is needed for linear model, so we load the model again
load(modelname);
om = model.omega;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(1:Nmod).*om(:)');
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = PHI_L2';

nz = size(p,1);
n = Nmod;

% State-space (continuous time) matrices
model.A = [zeros(size(K)), ones(size(K));-M\K -M\D];
model.B = [0; M\Fex1];
model.C = [PHI_L2 zeros(size(PHI_L2))];
model.D = 0;
model.nlcof = struct('power', p, 'coef', E);

%% multisine, using time domain formulation

upsamp = 1;  % Upsampling factor
R  = 4;           % Realizations. (one for validation and one for testing)
P  = 8;           % Periods, we need to ensure steady state
mds = 1;
for mds=[2 3 123]
switch mds
    case 1
        f1 = 250;          % low freq
        f2 = 500;        % high freq
    case 2
        f1 = 1350;          % low freq
        f2 = 1600;        % high freq
    case 3
        f1 = 3400;          % low freq
        f2 = 3650;        % high freq
    case 123
        f1 = [250 1350 3400];
        f2 = [500 1600 3650];
end
N  = 1e3;         % freq points
f0 = (f2(1)-f1(1))/N;
A = 0.01;  % Signal RMS amplitude
Alevels = [0.01 0.05 0.10 0.15 0.20 0.25];

for A=Alevels(1:end)
    fprintf('A = %f\n', A);
%     Nt = 2^15;      % Time per cycle  (2^13 for 4096; 2^15 for 16384)
%     fs = Nt*f0;     % Samping frequency
    
%     fs = 2^12;
    fs = 2^14;
    Nt = fs/f0;

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

    u    = zeros(Nt,P,R);
    y    = zeros(Nt,P,R,n);
    ydot = zeros(Nt,P,R,n);
    yout    = zeros(Nt,P,R);
    ydout = zeros(Nt,P,R);
    tic
    MS = cell(R, 1);
    for r=1:R
        % multisine force signal
        [fex, MS{r}] = multisine_multi(f1, f2, N, A, [], [], r);

        par = struct('M',M,'C',D,'K',K,'p',p,'E',E,'fex',fex, 'amp', Fex1);
%         [tout,Y] = ode45(@(t,y) odesys(t,y, par), t,[q0;u0]);
        Y = ode8(@(t,y) odesys(t,y, par), tint, [q0;u0]);
        Yout = Y(:,1:n)*PHI_L2';
        Ydout = Y(:,n+(1:n))*PHI_L2';

        u(:,:,r) = reshape(fex(t'), [Nt,P]);
        if upsamp > 1
            ytmp = dsample(reshape(Y(:,1:n), [Ntint,Pint,1,n]), upsamp);
            y(:,:,r,:) = ytmp;
            ytmp = dsample(reshape(Y(:,n+1:end), [Ntint,Pint,1,n]), upsamp);
            ydot(:,:,r,:) = ytmp;

            ytmp = dsample(reshape(Yout, [Ntint,Pint,1]), upsamp);
            yout(:,:,r) = ytmp;
            ytmp = dsample(reshape(Ydout, [Ntint,Pint,1]), upsamp);
            ydout(:,:,r) = ytmp;
        else
            y(:,:,r,:) = reshape(Y(:,1:n), [Nt,P,n]);
            ydot(:,:,r,:) = reshape(Y(:,n+1:end), [Nt,P,n]);
            
            yout(:,:,r) = reshape(Yout, [Nt, P]);
            ydout(:,:,r) = reshape(Ydout, [Nt, P]);
        end
    end
    disp(['ode45 with multisine in time domain required ' num2str(toc) ' s.']);

    save(sprintf('data/ode45_multisine_A%.2f_F%d_mds%s.mat',A,fs,sprintf('%d',mds)),'u','y','ydot','yout','ydout','f1','f2','fs','freq',...
        't','A','PHI_L2', 'MS', 'model', 'Nt','f0')

    %% show time series
%     A = 0.01;
    sprintf('data/ode45_multisine_A%.2f_F%d.mat',A,fs)

    load(sprintf('data/ode45_multisine_A%.2f_F%d_mds%s.mat',A,fs,sprintf('%d',mds)), 'yout')
    
    r = 1;
%     Y = PHI_L2*reshape(y(:,:,r,:),[],n)';
    Y = reshape(yout(:,:,r), [], 1)';

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
end