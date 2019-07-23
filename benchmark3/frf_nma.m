%========================================================================
% DESCRIPTION: 
% Investigation of a curved beam considering geometric nonlinearities.
%========================================================================
clear; clc;
close all;
srcpath = '../src/nlvib';
addpath(genpath(srcpath));

srcpath = '../src/matlab';
addpath(genpath(srcpath));

savedata = true;

%% Define system

% Fundamental parameters
Dmod = [.38 .09 .08]*.01; % first, third, fifth bending modes of flat beam
Nmod = 3;
setup = 'New_Design_Steel';

thickness = .001;
R=3;

[L,rho,E,~,PHI,~,gam] = beams_for_everyone(setup,Nmod*2-1,thickness);
PHI_L2 = PHI(L/2);
PHI_L2 = PHI_L2(1:2:end);
gam = gam(1:2:end);

% load nonlinear coefficients (can be found e.g. via IC-method)
modelname  = ['data/','beam_msh_80_4_1_3t_steel_',...
    num2str(thickness*1000),'mm_R',num2str(R),'m.mat'];
[p, E] = nlcoeff(modelname, Nmod);

% om is needed for linear model, so we load the model again
load(modelname);
om = model.omega;

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*Dmod(:).*om(:));
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = gam;

% Define oscillator as system with polynomial stiffness nonlinearities
oscillator = System_with_PolynomialStiffnessNonlinearity(M,D,K,p,E,Fex1);
% Number of degrees of freedom
ndof = oscillator.n;

% Linear modal analysis
[PHI_lin,OM2] = eig(oscillator.K,oscillator.M);
[om_lin,ind] = sort(sqrt(diag(OM2)));
PHI_lin = PHI_lin(:,ind);

%% HB settings

% Excitation levels
exc_lev = [10,30,50];
Nt = 1024;

for imod=[1,2,3]

switch imod
    case 1
        % start and end frequency in rad/s
        Om_s = 0;
        Om_e = 3*om_lin(1);
    case 2
        Om_s = 1200*2*pi;
        Om_e = 1600*2*pi;
    case 3
        Om_s = 3200*2*pi;
        Om_e = 3800*2*pi;
end


%% Convergence study for number of harmonics
% number of harmonics
Hs = [1:1:40];

% Set excitation level
oscillator.Fex1 = Fex1*exc_lev(end);
Xc = cell(size(Hs));
PkPs = zeros(length(Hs), 2);
for h = 1:length(Hs)
    Nh = Hs(h);
    
    % Initial guess (solution of underlying linear system)
    Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
    y0 = zeros((2*Nh+1)*length(Q1),1);
    y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
    
    ds = 50; % -> better for exc_lev = 50
    Dscale = [1e-6*ones(length(y0),1);(Om_s+Om_e)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full',...
        'stepmax',1e4);

    X = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,Nh,Nt,'frf'),...
        Om_s,Om_e,ds,Sopt);
    
    % calculate 90 degree phase for current mode
    % Sc = [freq, amp, phase]
    Sc = [X(end, :);
        sqrt([1 0.5*ones(1,2*Nh)]*X(imod:ndof:end-1,:).^2);
        atan2d(-X(2*ndof+imod,:), X(ndof+imod,:))]';
    PkPs(h,:) = interp1(Sc(:,3), Sc(:,1:2), -90, 'pchip');
    Xc{h} = X;
    
    fprintf('%d/%d Done.\n', h, length(Hs)) 
end

% Convergence Criterion
errmax = 0.01*1e-2;
Errs = abs(PkPs-PkPs(end,:))./PkPs;
ihconv = find(max(Errs,[],2)<=errmax, 1 );
Nhconv = Hs(ihconv);

save(sprintf('data/HConvDat_M%d.mat',imod), 'Xc', 'PkPs', 'Errs',...
    'errmax', 'Nhconv', 'ihconv','Hs','exc_lev')


%% Compute frequency response using harmonic balance
Nh = 7;
Nt=2*3*Nh+1;

X = cell(size(exc_lev));
for iex=1:length(exc_lev)
    % Set excitation level
    oscillator.Fex1 = Fex1*exc_lev(iex);
    
    % Initial guess (solution of underlying linear system)
    Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
    y0 = zeros((2*Nh+1)*length(Q1),1);
    y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];
    
    % Solve and continue w.r.t. Om
    ds = 50; % -> better for exc_lev = 50
        
    Dscale = [1e-6*ones(length(y0),1);(Om_s+Om_e)/2];
    Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full','stepmax',1e4);
    X{iex} = solve_and_continue(y0,...
        @(X) HB_residual(X,oscillator,Nh,Nt,'frf'),...
        Om_s,Om_e,ds,Sopt);

end

% create struct from varible names. Short hacky way.
frfdata = ws2struct('X','ds','Nh','Nt','exc_lev','oscillator','ndof'...
    ,'Dmod','Nmod','om','gam','thickness','setup','modelname','R','PHI_L2');
if savedata
    save(sprintf('data/frf_M%d.mat',imod),'-struct','frfdata')
end

%% NMA

Nh=7;
Nt=2*3*Nh+1;

log10a_s = -8;    % start vibration level (log10 of modal mass)
log10a_e = -3.2;  % end vibration level (log10 of modal mass)
inorm = 1;        % coordinate for phase normalization

oscillator.Fex1 = zeros(size(oscillator.Fex1));

% Initial guess vector y0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
om = om_lin(imod); phi = PHI_lin(:,imod);
Psi = zeros((2*Nh+1)*Nmod,1);
Psi(Nmod+(1:Nmod)) = phi;
x0 = [Psi;om;0];

% set eps=1 to avoid step-refinement due to large residual. Prevent turning
ds      = .1;
Sopt    = struct('Dscale',[1e-6*ones(size(x0,1)-2,1);1;1e-1;1],...
    'dynamicDscale',1,'stepmax',5e4,'eps',1);
[X,Solinfo,Sol] = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,Nh,Nt,'nma',inorm),...
    log10a_s,log10a_e,ds, Sopt);


nmadata = ws2struct('Solinfo','Sol','X','ds','Nh','Nt','ndof','imod',...
    'oscillator','Dmod','Nmod','om','gam','thickness','setup',...
    'modelname','R','PHI_L2','Sopt');
if savedata
    save(sprintf('data/nma_M%d.mat',imod),'-struct','nmadata')
end

end
%% plot data
%hb_plot;
