clear; clc;
close all; 
addpath('00_SRC');
addpath('00_SRC/MechanicalSystems');
set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

%% Define system

% Fundamental parameters
Dmod = [.38 .12 .09 .08 .08]*.01;
Nmod = 5;
setup = 'New_Design_Steel';
thickness = .001;
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod,thickness);
PHI_L_2 = PHI(L/2);

% load nonlinear coefficients (can be found e.g. analytically)
load(['beam_New_Design_Steel_analytical_5t_' num2str(thickness*1000) 'mm.mat'])
model.b=b;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(:).*om(:));
K = diag(om.^2);

% polynomial terms
p_quad = zeros(sum(1:Nmod),Nmod);
p_cub = zeros(sum(cumsum(1:Nmod)),Nmod);
ctr_quad = 1; ctr_cub = 1;

for jj = 1:Nmod
    for kk = jj:Nmod
        % quadratic terms
        p_quad(ctr_quad,jj) = p_quad(ctr_quad,jj)+1;
        p_quad(ctr_quad,kk) = p_quad(ctr_quad,kk)+1;
        ctr_quad = ctr_quad+1;
        for ll = kk:Nmod
            % cubic terms
            p_cub(ctr_cub,jj) = p_cub(ctr_cub,jj)+1;
            p_cub(ctr_cub,kk) = p_cub(ctr_cub,kk)+1;
            p_cub(ctr_cub,ll) = p_cub(ctr_cub,ll)+1;
            ctr_cub = ctr_cub+1;
        end
    end
end

p = [p_cub];

% coefficients
E=zeros(sum(cumsum(1:Nmod)),Nmod);

for rr = 1:Nmod
    ctr = 1;
%         for jj = 1:Nmod
%             for kk = jj:Nmod
%                 % quadratic coeffs
%                 E(ctr,rr) = model.a(jj,kk,rr);
%                 ctr = ctr+1;
%             end
%         end
%         ctr = 1;
    for jj = 1:Nmod
        for kk = jj:Nmod
            for ll = kk:Nmod
                % cubic coeffs
                E(ctr,rr) = model.b(jj,kk,ll,rr);
                ctr = ctr+1;
            end
        end
    end
end

% Fundamental harmonic of external forcing
Fex1 = gam;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);

% Number of degrees of freedom
n = oscillator.n;

%% Frequency Response
analysis = 'FRF';
H = 7;              % harmonic order
N=2*3*H+1;
Ntd = 1e3;

% Analysis parameters
Om_e = om(1)*.1;      % start frequency
Om_s = 5*om(1);     % end frequency

% Excitation levels
exc_lev = [1 10 100 1000];
Om = cell(size(exc_lev));
a_w_L_2 = Om;
ph_all = Om;
amp_all = Om;

for iex=1:length(exc_lev)
    % Set excitation level
    oscillator.Fex1 = Fex1*exc_lev(iex);

    % Initial guess (solution of underlying linear system)
    Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
    y0 = zeros((2*H+1)*length(Q1),1);
    y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
    qscl = max(abs((-om(1)^2*M + 1i*om(1)*D + K)\oscillator.Fex1));
    
    % Solve and continue w.r.t. Om
    ds = 50; % -> better for exc_lev = 50
        
    Dscale = [1e-6*ones(length(y0),1);(Om_s+Om_e)/2];
    if iex<=2
        Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4,...
            'dsmax',100);
    else
        Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4,...
            'dsmax',250);
    end
    X = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,H,N,analysis),...
        Om_s,Om_e,ds,Sopt);
    
    % Interpret solver output
    Om{iex} = X(end,:);
    Q_HB = X(1:end-1,:);
    
    % Define amplitude as magnitude of the fundamental harmonic of the
    % first coordinate's displacement
    tau = linspace(0,2*pi,Ntd+1); tau = tau(1:end-1);    
    w_L_2_sum = zeros(Ntd,length(Om{iex}));                                 % lines = time steps, columns = continuation points
    a_q = zeros(1, length(Om{iex}));
    for k = 1:n
        Qc{k} = [Q_HB(k,:);Q_HB(n+k:2*n:end,:)-1i*Q_HB(2*n+k:2*n:end,:)];   % get complex coefficients
        w_L_2{k} = PHI_L_2(k)*real(exp(1i*tau(:)*(0:H))*Qc{k});             % get displacement at center caused by each mode in time domain
        w_L_2_sum = [w_L_2_sum + w_L_2{k}];                                 % sum up to actual displacement at center in time domain
        
        q{k} = real(exp(1i*tau(:)*(0:H))*Qc{k});
        a_q(k,:) = (max((q{k}))-min((q{k})))/2;
    end
    
    a_w_L_2{iex} = (max((w_L_2_sum))-min((w_L_2_sum)))/2;                         % compute peak to peak amplitude
    clear w_L_2
    
    % Calculate Response Phase
    amp_all{iex} = zeros(n+1,length(Om{iex}));
    ph_all{iex} = zeros(n+1,length(Om{iex}));
    for k = 1:n
        Qc{k} = [Q_HB(k,:);Q_HB(n+k:2*n:end,:)-1i*Q_HB(2*n+k:2*n:end,:)];   % get complex coefficients
        ph_all{iex}(k, :)  = rad2deg(angle(Qc{k}(2,:)));
        amp_all{iex}(k, :) = range(real(exp(1i*tau(:)*(0:H))*Qc{k}))/2;
    end
    Qc_L_2 = kron(eye(2*H+1), PHI_L_2)*Q_HB;
    ph_all{iex}(n+1,:)  = atan2d(-Qc_L_2(3,:), Qc_L_2(2,:));
    amp_all{iex}(n+1,:) = a_w_L_2{iex};
end

%% NMA
H=7;
N=2*3*H+1;
Ntd = 1e3;

% Linear modal analysis
[PHI_lin,OM2] = eig(oscillator.K,oscillator.M);
[om_lin,ind] = sort(sqrt(diag(OM2)));
PHI_lin = PHI_lin(:,ind);

analysis = 'NMA';

oscillator.Fex1 = Fex1*0;

imod = 1;           % mode to be analyzed
log10a_s = -8;    % start vibration level (log10 of modal mass)
log10a_e = -3.2;       % end vibration level (log10 of modal mass)
inorm = 1;          % coordinate for phase normalization

% Initial guess vector y0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
om = om_lin(imod); phi = PHI_lin(:,imod);
Psi = zeros((2*H+1)*Nmod,1);
Psi(Nmod+(1:Nmod)) = phi;
x0 = [Psi;om;0];

ds      = .1;
Sopt    = struct('Dscale',[1e-6*ones(size(x0,1)-2,1);1;1e-1;1],...
    'dynamicDscale',1,'stepmax',5e4);
[X_HB,Solinfo,Sol] = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,H,N,analysis,inorm),...
    log10a_s,log10a_e,ds, Sopt);

Psi_HB = X_HB(1:end-3,:);
om_HB = X_HB(end-2,:);
del_HB = X_HB(end-1,:);
log10a_NMA = X_HB(end,:);
a_NMA = 10.^log10a_NMA;
Q_HB = Psi_HB.*repmat(a_NMA,size(Psi_HB,1),1);

% Define amplitude as magnitude of the fundamental harmonic of the
% first coordinate's displacement
tau = linspace(0,2*pi,Ntd+1); tau = tau(1:end-1);
w_L_2_NMA_sum = zeros(Ntd,length(om_HB));                                   % lines = time steps, columns = continuation points
for k = 1:n
    Qc_NMA{k} = [Q_HB(k,:);Q_HB(n+k:2*n:end,:)-1i*Q_HB(2*n+k:2*n:end,:)];   % get complex coefficients
    w_L_2_NMA{k} = PHI_L_2(k)*real(exp(1i*tau(:)*(0:H))*Qc_NMA{k});         % get displacement at center caused by each mode in time domain
    w_L_2_NMA_sum = [w_L_2_NMA_sum + w_L_2_NMA{k}];                         % sum up to actual displacement at center in time domain
    
    q_NMA{k} = real(exp(1i*tau(:)*(0:H))*Qc_NMA{k});
%     a_q_NMA(k,:) = (max((q_NMA{k}))-min((q_NMA{k})))/2;
    a_q_NMA(k,:) = range(q_NMA{k})/2;
end

a_w_L_2_NMA= (max((w_L_2_NMA_sum))-min((w_L_2_NMA_sum)))/2;                 % compute peak to peak amplitude

%% Plotting without interpolations
figure(10)
clf()
colos = distinguishable_colors(length(exc_lev),'k');

aa = gobjects(length(exc_lev)+1, 1);
aa(1) = semilogy(om_HB/(2*pi), a_w_L_2_NMA, 'k+-', 'linewidth', 1.5); hold on
legend(aa(1), 'EPMC')

p2 = (om_HB.^2-2*(del_HB.*om_HB).^2)';
om4 = (om_HB.^4)';
Phi_HB = Psi_HB(Nmod+(1:Nmod),:)-1j*Psi_HB(2*Nmod+(1:Nmod),:);
Fsc = (abs(Phi_HB'*Fex1)./a_NMA').^2;
mAmps = a_NMA.*range(real(exp(1j*tau(:))*(PHI_L_2*Phi_HB)))/2;

ivs = 10;
for k=1:length(exc_lev)
    aa(k+1) = semilogy(Om{k}/(2*pi), a_w_L_2{k}, '-', 'Color', colos(k,:), 'Linewidth', 2);
    legend(aa(k+1), sprintf('Fa = %.0f', exc_lev(k)));
    
    om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*exc_lev(k)^2));
    om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*exc_lev(k)^2));
    
    ris1 = find(imag(om1)==0);  % real indices
    ris2 = find(imag(om2)==0);  % real indices
    semilogy(om1(ris1)/(2*pi), mAmps(ris1), '--', 'Color', colos(k,:)); hold on
    semilogy(om1(ris1(1:ivs:end))/(2*pi), mAmps(ris1(1:ivs:end)),...
        'o', 'Color', colos(k,:), 'MarkerFaceColor', colos(k,:)); hold on
    semilogy(om2(ris2)/(2*pi), mAmps(ris2), '--', 'Color', colos(k,:)); hold on
    semilogy(om2(ris2(1:ivs:end))/(2*pi), mAmps(ris2(1:ivs:end)),...
        'o', 'Color', colos(k,:), 'MarkerFaceColor', colos(k,:)); hold on
    
    legend('hide')
end
xlim([0 1400])

legend(aa(1:end), 'Location', 'best')
xlabel('Forcing Frequency (Hz)')
ylabel('Peak-Peak Amplitude at L/2 (m)')
% print('./FIGS/NMROMCCFB.eps', '-depsc')

xlim([350 650])
ylim([5e-5 5e-3])
legend('hide')
% print('./FIGS/NMROMCCCB_ZOOM.eps', '-depsc')