clear; clc;
close all;

srcpath = '../../src/nlvib';
addpath(genpath(srcpath));
srcpath = '../';
addpath(genpath(srcpath));

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0,'DefaultLegendInterpreter', 'latex');

%% Define system

% Fundamental parameters
Dmod = [.38]*.01;
Nmod = 1;
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

%% Compute frequency response using harmonic balance
analysis = 'FRF';
i = 1;
for H = 1:2:40              % harmonic order
N=2*3*H+1;
Ntd = 1e3;

% Analysis parameters
Om_s = om(1)*.2;      % start frequency
Om_e = 3*om(1);     % end frequency

% Excitation levels
exc_lev = 100;
Om = cell(size(exc_lev));

% Set excitation level
oscillator.Fex1 = Fex1*exc_lev;

% Initial guess (solution of underlying linear system)
Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
y0 = zeros((2*H+1)*length(Q1),1);
y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
qscl = max(abs((-om(1)^2*M + 1i*om(1)*D + K)\oscillator.Fex1));

% Solve and continue w.r.t. Om
ds = 50; % -> better for exc_lev = 50

Dscale = [1e-6*ones(length(y0),1);(Om_s+Om_e)/2];
Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4);
X = solve_and_continue(y0,...
    @(X) HB_residual(X,oscillator,H,N,analysis),...
    Om_s,Om_e,ds,Sopt);

% Interpret solver output
Om = X(end,:);
Q_HB = X(1:end-1,:);

% Define amplitude as magnitude of the fundamental harmonic of the
% first coordinate's displacement
w_L_2_sum = zeros(2*H+1,length(Om));
for k = 1:n
    Qc{k} = [Q_HB(k,:);Q_HB(n+k:n:end,:)]; 
    w_L_2{k} = PHI_L_2(k)*Qc{k};                  % get displacement at center caused by each mode in time domain
    w_L_2_sum = [w_L_2_sum + w_L_2{k}];
end
a_w_L_2 = sqrt([1 0.5*ones(1,2*H)]*w_L_2_sum.^2); % compute amplitude
phase = atan2d(-w_L_2_sum(3,:),w_L_2_sum(2,:));
res(i) = interp1(phase,Om/2/pi,-90); % compute resonance freq
peak_amp(i) = interp1(Om/2/pi,a_w_L_2*1000,res(i)); % compute resonance amplitude
i = i+1;
end

Harmonics = 1:2:40;
tolerence = ones(length(Harmonics),1)*0.1;
amp_err = abs(peak_amp - peak_amp(end))/peak_amp(end)*100; % resonance amplitude error
freq_err = abs(res - res(end))/res(end)*100; % resonance freq error

save HarmonicError.mat




