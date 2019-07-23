clc
clear variables

addpath('../src/pnlss');
addpath('../src/matlab');
%% Define system

% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.01;
Nmod = 1;
setup = 'New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
fname = ['beam_New_Design_Steel_analytical_5t_' ...
    num2str(thickness*1000) 'mm.mat'];
[p, E] = nlcoeff(fname, Nmod);
E = 0;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(1:Nmod).*om(1:Nmod));
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = gam;


%% state space

fs_init = 2^12;
fs = fs_init;

% Continuous time model
ctmodel.A = [0 1;
            -M\K -M\D];
ctmodel.B = [0; M\Fex1];
ctmodel.C = [PHI_L2 0];
ctmodel.D = [0];
% xpowers: each row gives the power of [x1, x2, u], which is then
% multiplied with corresponding coefficient in E.
% E ∈ [n, nx], xpowers ∈ [nx, n+1]
ctmodel.xpowers = [3 0 0];
ctmodel.E = [0; -M\E];
ctmodel.F = 0;
ctmodel.ypowers = [3 0 0];

% analytical discretization for A,B
dtmodel = ctmodel;
dtmodel.A = expm(ctmodel.A/fs);
dtmodel.B = ctmodel.A\(dtmodel.A-eye(2))*ctmodel.B;
dtmodel.E = ctmodel.E/fs;
sys = dtmodel;

% parameters for PNLSS. Initialize as linear. We change this later
% nx/ny is not needed for time simulation
nx = []; ny = []; T1 = 0; T2= 0;
model = fCreateNLSSmodel(sys.A,sys.B,sys.C,sys.D,nx,ny,T1,T2);
model.E = dtmodel.E;
model.F = dtmodel.F;
model.xpowers = dtmodel.xpowers;
model.ypowers = dtmodel.ypowers;

% % transient periods to simulate and remove to ensure steady state
% NTrans = 10*Nt;  % Add 10 periods before the start
% % Number of transient samples and starting indices of each realization
% T1 = [NTrans 1+(0:Nt*P:(R-1)*Nt*P)]; 
% model.T1 = T1;
% model.T1 = NTrans;

%% simulate model -- From here is new!
% direction to sweep
dir = 1; % or -1

% We see each increment in frequency as a new realization.
% Vector of excited frequencies. In Hz!
Om_vec = linspace(200,350,25);
Om_vec = [linspace(200, 260, 10) linspace(260, 270, 20) linspace(270, 350, 10)];

if dir == -1
    Om_vec = fliplr(Om_vec);
end
R = length(Om_vec);
P = 20;

% amplitudes to loop over
Avec = 80; %[10,40,80,100];
% Avec = [10 40 80 100];

% periods to calculate max amp over.
nper = 5;

%% Simulate with constant fs, non-integer periods
% this is the only way to simulate the nolinear sys. Slow!

Nt = 1e3;
Ptrans = 150;
nsim = fs/Nt;
T = linspace(0, (P+Ptrans)/nsim, Nt*(P+Ptrans)+1)'; T(end) = [];

a_max = zeros(R,length(Avec));

a_rms = zeros(R,length(Avec));
ph_rs = zeros(R,length(Avec));

a_rms_R = zeros(R,length(Avec));
ph_rs_R = zeros(R,length(Avec));

j = 1;
Ntint = 2^4;
Yint = zeros(Nt,1);
y = zeros(Ntint*P,R);
for A = Avec
    X0 = zeros(length(model.A),1);
    for i = 1:R
        Om = Om_vec(i);
        u = A*sin(2*pi*Om*T);
        [Y, ~, X] = lsim(ss(model.A,model.B,model.C,model.D,-1),u,[],X0);
%         [Y, X] = fFilterNLSS(model,u,X0); % Modeled output signal
        % [Y, ~, X] = lsim(ss(ctmodel.A,ctmodel.B,ctmodel.C,ctmodel.D),u,T,X0);
        X0 = X(end,:);
        ymax = max(Y(end-nper*Nt+1:end));
        ymin = min(Y(end-nper*Nt+1:end));
        a_max(i,j) = abs(0.5*(ymax-ymin));
        
        % REGRESSION APPROACH
        YR = [ones(size(T(end-(P*Nt)+1:end))) cos(2*pi*Om*T(end-(P*Nt)+1:end)) sin(2*pi*Om*T(end-(P*Nt)+1:end)) cos(2*2*pi*Om*T(end-(P*Nt)+1:end)) sin(2*2*pi*Om*T(end-(P*Nt)+1:end))];
        TH = YR\Y(end-(P*Nt)+1:end);        
        a_rms_R(i,j) = sqrt(TH(1)^2 + 0.5*sum(TH(2:end).^2));
        ph_rs_R(i,j) = rad2deg(angle((TH(2)-1j*TH(3))./Ufint(2)));
        
        % INTERPOLATION APPROACH
        % Interpolating response from grid T that is sampled at the fixed
        % sampling rate "fs" to grid Tint that is sampled at the sampling
        % rate Ntint*Om
        Pint = floor(T(end)*Om);
        Tint = linspace(0, Pint/Om, Ntint*Pint+1)';  Tint(end) = [];
        Yint = interp1(T, Y, Tint, 'pchip');

        % Removal of transients on interpolated response data
        Yint = Yint(end-(P*Ntint)+1:end);
        y(:, i) = Yint;
        
        Yrint = mean(reshape(Yint, Ntint, P), 2);
        Urint = A*sin(2*pi*Om*Tint(1:Ntint));
        
        Ufint = fft(urint)/(Ntint/2); Ufint(1) = Ufint(1)/2;
        Yfint = fft(Yrint)/(Ntint/2); Yfint(1) = Yfint(1)/2;
        
        a_rms(i,j) = sqrt(Yfint(1)^2 + 0.5*vecnorm(Yfint(2:Ntint/2)).^2);
        ph_rs(i,j) = rad2deg(angle(Yfint(2)/Ufint(2)));
%         ph_rs(i,j) = rad2deg(angle(Yfint(2)));
        if sum(any(isnan(Y)))
            fprintf('Error: simulation exploded. Try increase fs. r:%d\n',i)
            break % don't quit, we still want to save data.
        end
    end
    j = j+1;
    fprintf('A: %g\n',A)
end

save(sprintf('data/stepped_sweep_dir%d',dir),'Om_vec','a_max')

% % plot max amplitudes -- poor mans frf
% figure
% plot(Om_vec, a_max, 'x-')

% xlabel('Frequency (Hz)')
% ylabel('Max amplitude (m)')

% plot rms amplitudes -- rich mans frf
figure(1)
clf()
plot(Om_vec, a_rms, '.-', Om_vec, a_rms_R, 'x-')
xlabel('Frequency (Hz)')
ylabel('RMS amplitude (m)')

% plot phase
figure(2)
clf()
plot(Om_vec, ph_rs, '.-', Om_vec, ph_rs_R, 'x-')
xlabel('Frequency (Hz)')
ylabel('Response relative phase (degs)')

% % Plot time signal
% t = linspace(0, (P)/nsim, Nt*(P)+1); t(end) = [];
% figure
% hold on
% for i = 1:R
%     plot(t, y(:,i))
% end
% xlabel('Time (s)')
% ylabel('Amplitude (m)')


%% Simulate with changing fs, keep Nt and Om constant. Integer periods
% this is the fastest way to simulate the linear system

Nt = 2^12;                          % points per period
t = zeros(R, Nt*P+1);  % end time depends on forcing freq
for i = 1:R
    t(i,:) = linspace(0, P/Om_vec(i), Nt*P+1);  % If Om (rad/s), multiply end time by 2*pi
end
t(:,end) = [];

% transient periods to simulate and remove to ensure steady state
NTrans = 10*Nt;  % Add 10 periods before the start
% Number of transient samples and starting indices of each realization
T1 = [NTrans 1+(0:Nt*P:(R-1)*Nt*P)]; 
model.T1 = T1;
model.T1 = NTrans;

% amplitudes to loop over
Avec = 80; %[10,40,80,100];
% Avec = [10 40 80 100];
a_max = zeros(length(Om_vec),length(Avec));
a_rms = zeros(length(Om_vec),length(Avec));
ph_rs = zeros(length(Om_vec),length(Avec));
j = 1;
for A = Avec
    X0 = zeros(length(model.A),1);
    y = zeros(Nt*P,R);
    for i = 1:R
        Om = Om_vec(i);
        Ptrans = 150;
        T = linspace(0, (P+Ptrans)/Om, Nt*(P+Ptrans)+1); T(end) = [];
        u = A*sin(2*pi*Om*T);
        fs = Nt*Om;
        [Y, ~, X] = lsim(c2d(ss(ctmodel.A,ctmodel.B,ctmodel.C,ctmodel.D),1/fs),u,[],X0);
%         [y, ~, X] = lsim(ss(ctmodel.A,ctmodel.B,ctmodel.C,ctmodel.D),u,T,X0);
%         [Y, ~, X] = lsim(ss(model.A,model.B,model.C,model.D,1/fs),u,[],X0);
%         [y, X] = fFilterNLSS(model,u,X0); % Modeled output signal
        X0 = X(end,:);
        Y = Y(Ptrans*Nt+1:end);
        ymax = max(Y(end-nper*Nt+1:end));
        ymin = min(Y(end-nper*Nt+1:end));
        a_max(i,j) = abs(0.5*(ymax-ymin));
        
        % FFT on last period
        U = u(Ptrans*Nt+1:end)';
        yf = fft(Y(end-Nt+1:end))/(Nt/2);  yf(1) = yf(1)/2;
        uf = fft(U(end-Nt+1:end))/(Nt/2); uf(1) = uf(1)/2;
        %         a_rms(i,j) = sqrt(sum(Y.^2)/fs*Om/P);  % trapezoidal integration; inaccurate
        a_rms(i,j) = sqrt(yf(1)^2 + 0.5*vecnorm(yf(2:Nt/2)).^2);  % might as well use freq dom.
        ph_rs(i,j) = rad2deg(angle(yf(2)/uf(2)));
        y(:,i) = Y;
    end
    j = j+1;
    fprintf('A: %g\n',A)
end

% plot rms amplitudes -- rich mans frf
figure
plot(Om_vec, a_rms, 'x-')
xlabel('Frequency (Hz)')
ylabel('RMS amplitude (m)')

figure
plot(Om_vec, a_max, 'x-')
xlabel('Frequency (Hz)')
ylabel('Max amplitude (m)')

% plot phases
figure
plot(Om_vec, ph_rs, 'x-')
xlabel('Response relative phase (degs)')
ylabel('Max amplitude (m)')

% Plot time signal
y1 = reshape(y, [Nt*P,R]);
figure
hold on
for i = 1:R
    plot(t(i,:), y1(:,i))
end
xlabel('Time (s)')
ylabel('Amplitude (m)')

% Check periodicity
yper = reshape(y, [Nt,P,R]);
r = 1;
figure;
per = (yper(:,1:end-1,r)-yper(:,end,r)) / rms(yper(:,end,r));
plot(t(r,1:Nt*(P-1)),db(per(:)),'k-')
% indicate periods
h1 = vline(t(r,(1:Nt:(P-1)*Nt)),'--g'); set([h1],'LineWidth',0.5)
xlabel('time (s)')
ylabel('Relative error to last period (dB)')
title([num2str(Nt) ' samples per period'])

%% Simulate integer periods. Constant fs. Change Nt and Om accordingly
% To keep the sampling frequency fixed, while changing Om and still only
% simulating a fixed number of periods; Nt is calculated for each Om,
% rounded and then Om is recalculated.
% Nt can be very very low(if fs is), we need more periods for steady state.
fs = fs_init;
dt = 1/fs;
Nt_vec = fix(1./(Om_vec*dt));
Om_vec = 1./(Nt_vec*dt);
j = 1;
for A = Avec
    X0 = zeros(length(model.A),1);
    y = cell(length(Om_vec),1);
    t = cell(length(Om_vec),1);
    for i = 1:length(Om_vec)
        Om = Om_vec(i);
        Nt = Nt_vec(i);
        Ptrans = 500;
        T = linspace(0, (P+Ptrans)/Om, Nt*(P+Ptrans)+1); T(end) = [];
        u = A*sin(2*pi*Om*T);
        [Y, ~, X] = lsim(ss(model.A,model.B,model.C,model.D,-1),u,[],X0);
        X0 = X(end,:);
        ymax = max(Y(end-nper*Nt+1:end));
        ymin = min(Y(end-nper*Nt+1:end));
        a_max(i,j) = abs(0.5*(ymax-ymin));
        y{i} = Y;
        t{i} = T;
    end
    j = j+1;
    fprintf('A: %g\n',A)
end

% plot max amplitudes -- poor mans frf
figure
plot(Om_vec, a_max, 'x-')
xlabel('Frequency (Hz)')
ylabel('Max amplitude (m)')

% Plot time signal
figure
hold on
for i = 1:length(Om_vec)
    plot(t{i}, y{i})
end
xlabel('Time (s)')
ylabel('Amplitude (m)')
