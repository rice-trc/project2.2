clear all
clc

benchmark = 3;
Nt = 1024;

setup = 'New_Design_Steel';
switch benchmark
    case 1
        Dmod = [.38]*.01;
        Nmod = 1;
        imods = [1];
        Hs = [1:2:40 40];  % only unenven harmonics
        exc_lev = [10,40,60,80,100];
        % start and end vibration level (log10 of modal mass)
        log10a_ss = [1]*(-5);
        log10a_ee = [1]*(-3.2);
        Nhs = 3;
    case 2
        Dmod = [.38 .12 .09 .08 .08]*.01;
        Nmod = 5;
        thickness = .001;
        % load nonlinear coefficients (can be found e.g. analytically)
        modelname = ['data/beam_New_Design_Steel_analytical_5t_' ...
            num2str(thickness*1000) 'mm.mat'];

        imods = [1,2,3];
        Hs = [1:2:40 40];  % only unenven harmonics
        exc_lev = [10,40,60,80,100];
        % start and end vibration level (log10 of modal mass)
        log10a_ss = [1,1,1]*(-5);
        log10a_ee = [1,1,1]*(-3.2);
        Nhs = ones(lsize(imods))*7;
    case 3
        % Fundamental parameters
        Dmod = [.38 .09 .08]*.01; % first, third, fifth bending modes of flat beam
        Nmod = 3;
        thickness = .001;
        R=3;
        % load nonlinear coefficients (can be found e.g. via IC-method)
        modelname  = ['data/beam_msh_80_4_1_3t_steel_',...
            num2str(thickness*1000),'mm_R',num2str(R),'m.mat'];
        
        imods = [1,2,3];
        % Convergence study for number of harmonics
        Hs = [1:1:40];
        % Excitation levels
        exc_lev = [10,30,50];
        log10a_ss = [1,1,1]*(-8);
        log10a_ee = [1,1,1]*(-3.2);
        Nhs = ones(size(imods))*7;
end

%% load analytical parameters
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,Nmod*2-1,thickness);
PHI_L2 = PHI(L/2);

% load nl coefficient
[p, E] = nlcoeff(modelname, Nmod, benchmark);

if benchmark == 3
    PHI_L2 = PHI_L2(1:2:end);
    
    gam = gam(1:2:end);
    % om is needed for linear model, so we load the model directly
    load(modelname);
    om = model.omega;
    
end

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


%% hb settings
% start and end frequency in rad/s
switch benchmark
    case 1
        Om_ss = [om(1)*.2];
        Om_ee = [3*om(1)];
    case 2
        Om_ss = [om(1)*.2, om(2)*.75, om(3)*.75];
        Om_ee = [3*om(1), om(2)*1.20, om(3)*1.20];
    case 3
        Om_ss = [0, 1300*2*pi, 3400*2*pi];
        Om_ee = [3*om_lin(1), 1600*2*pi, 3700*2*pi];
end

for imod = imods

Om_s = Om_ss(imod);
Om_e = Om_ee(imod);

%% convergence analysis
% Set excitation level
oscillator.Fex1 = Fex1*exc_lev(end);
Nt = 1024;
Xc = cell(size(Hs));
PkPs = zeros(length(Hs), 2);
for h = 1:length(Hs)
    Nh = Hs(h);
    [X,Solinfo,Sol] = hb_frf(Om_s, Om_e, Nh, Nt,oscillator);
    
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

save(sprintf('data/b%d_HConvDat_M%d.mat',benchmark,imod),...,
    'Xc', 'PkPs', 'Errs', 'errmax', 'Nhconv', 'ihconv','Hs','exc_lev')

%% Compute frequency response using harmonic balance
Nh = Nhs(imod);
Nt=2*3*Nh+1;

X = cell(size(exc_lev));
for iex=1:length(exc_lev)
    % Set excitation level
    oscillator.Fex1 = Fex1*exc_lev(iex);
    [X{iex},Solinfo,Sol] = hb_frf(Om_s, Om_e, Nh, Nt,oscillator);
    
end
% create struct from varible names. Short hacky way.
frfdata = ws2struct('X','ds','Nh','Nt','exc_lev','oscillator','ndof'...
    ,'Dmod','Nmod','om','gam','thickness','setup','modelname','R','PHI_L2');
if savedata
    save(sprintf('data/b%d_frf_M%d.mat',benchmark,imod),'-struct','frfdata')
end

%% NMA

Nh = Nhs(imod);
Nt=2*3*Nh+1;
log10a_s = log10a_ss(imod)-8;
log10a_e = log10a_ee(imod)-3.2;

[X,Solinfo,Sol] = hb_nma(log10a_s,log10a_e, Nh, Nt,oscillator,...
    om_lin, PHI_lin);

nmadata = ws2struct('Solinfo','Sol','X','ds','Nh','Nt','ndof','imod',...
    'oscillator','Dmod','Nmod','om','gam','thickness','setup',...
    'modelname','R','PHI_L2','Sopt');
if savedata
    save(sprintf('data/nma_M%d.mat',imod),'-struct','nmadata')
end


end


function [X,Solinfo,Sol] = hb_frf(Om_s, Om_e, Nh, Nt,oscillator)

K = oscillator.K;
M = oscillator.M;
D = oscillator.D;

% Initial guess (solution of underlying linear system)
Q1 = (-Om_s^2*M + 1i*Om_s*D + K)\oscillator.Fex1;
y0 = zeros((2*Nh+1)*length(Q1),1);
y0(length(Q1)+(1:2*length(Q1))) = [real(Q1);-imag(Q1)];

ds = 50; % -> better for exc_lev = 50
Dscale = [1e-6*ones(length(y0),1);(Om_s+Om_e)/2];
Sopt = struct('Dscale',Dscale,'dynamicDscale',1,'jac','full',...
    'stepmax',1e4);

[X,Solinfo,Sol] = solve_and_continue(y0,...
    @(X) HB_residual(X,oscillator,Nh,Nt,'frf'),...
    Om_s,Om_e,ds,Sopt);

end

function [X,Solinfo,Sol] = hb_nma(log10a_s,log10a_e, Nh, Nt,oscillator,...
    om_lin, PHI_lin)


inorm = 1;        % coordinate for phase normalization
nmod = oscillator.n;

% Initial guess vector y0 = [Psi;om;del], where del is the modal
% damping ratio, estimate from underlying linear system
om = om_lin(imod); phi = PHI_lin(:,imod);
Psi = zeros((2*Nh+1)*nmod,1);
Psi(Nmod+(1:Nmod)) = phi;
x0 = [Psi;om;0];

% set eps=1 to avoid step-refinement due to large residual. Prevent turning
ds      = .1;
Sopt    = struct('Dscale',[1e-6*ones(size(x0,1)-2,1);1;1e-1;1],...
    'dynamicDscale',1,'stepmax',5e4,'eps',1);
[X,Solinfo,Sol] = solve_and_continue(x0,...
    @(X) HB_residual(X,oscillator,Nh,Nt,'nma',inorm),...
    log10a_s,log10a_e,ds, Sopt);

end

