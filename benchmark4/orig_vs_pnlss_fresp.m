clc
clear all

addpath('../src/mhbm_res/')
addpath('../src/matlab/')
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

%% System definition
len = 0.70;
hgt = 0.03;
thk = hgt;
E   = 185e9;
rho = 7830.0;
BCs = 'clamped-free';

Nn = 8;
beam = FE_EulerBernoulliBeam(len, hgt, thk, E, rho, BCs, Nn);
fdof = beam.n-1;
Fex1 = zeros(beam.n, 1);  Fex1(fdof) = 1;
% Nonlinearity
Nnl = 4;
dir = 'trans';
kt  = 1.3e6;
muN = 1;
add_nonlinear_attachment(beam, Nnl, dir, 'elasticdryfriction', ...
    'stiffness', kt, 'friction_limit_force', muN, ...
    'ishysteretic', true);
Nd = size(beam.M, 1);
%% Linearized limit cases
% Slipped
[Vsl, Dsl] = eig(beam.K, beam.M);
[Dsl, si] = sort(sqrt(diag(Dsl)));
Vsl = Vsl(:, si);  Vsl = Vsl./sqrt(diag(Vsl'*beam.M*Vsl)');

% Stuck
Knl = zeros(size(beam.M));
nli = find(beam.nonlinear_elements{1}.force_direction);
Knl(nli, nli) = kt;
Kst = beam.K + Knl;
[Vst, Dst] = eig(Kst, beam.M);
[Dst, si] = sort(sqrt(diag(Dst)));
Vst = Vst(:, si); Vst = Vst./sqrt(diag(Vst'*beam.M*Vst));

%% Rayleigh damping
% Desired
zs = [8e-3; 2e-3];
ab = [ones(length(zs),1) Dst(1:length(zs)).^2]\(2*zs.*Dst(1:length(zs)));

beam.D = ab(1)*beam.M + ab(2)*Kst;

Zetas = diag(Vst'*beam.D*Vst)./(2*Dst);

%% Parameters for HBM
Nt = 2^10;
imod = 1;  % Desired mode

switch imod
    case 1
        Fas = [0.01 0.05 0.1 0.2 0.3 0.6];

        Ws = 200;
        We = 500;
        ds = abs(We-Ws)/100;
        dsmax = abs(We-Ws)/50;
        xls = [20 100];
        yls = [1e-8 1e-4];
        
        Wsc = 70*2*pi;
        Wec = 55*2*pi;
        
        log10a_s = -7;
        log10a_e = -3.8;
        dl10a = 0.01;
        dl10amax = 0.05;
        
        Nhconv = 13;
    case 2
        Fas = [0.02 0.15 0.3 0.6 0.8 1.0];

        Ws = 2500;
        We = 1650;
        ds = abs(We-Ws)/100;
        dsmax = abs(We-Ws)/50;
        xls = [250 450];
        yls = [5e-9 1e-5];
        
        Wsc = 312*2*pi;
        Wec = 335*2*pi;
        
        log10a_s = -7.5;
        log10a_e = -3.8;
        dl10a = 0.0001;
        dl10amax = 0.05;
        
        Nhconv = 7;
    case 3
        Fas = [0.1 0.25 0.75 1.25 2.25 3.0];

        Ws = 5500;
        We = 5200;
        ds = abs(We-Ws)/100;
        dsmax = abs(We-Ws)/50; 
        xls = [800 900];
        yls = [3e-8 2e-5];
        
        Wsc = 840*2*pi;
        Wec = 850*2*pi;
        
        log10a_s = -7.5;
        log10a_e = -3.8;
        dl10a = 0.0001;
        dl10amax = 0.05;
        
        Nhconv = 3;
end

%% FRF
Nh = Nhconv;
Nhc = 2*Nh+1;
Dscale = [1e-4*ones(Nd*Nhc,1); (Dst(1)+Dsl(1))/2];

Xcont = cell(size(Fas));
Sols = cell(size(Fas));
Pks = zeros(length(Fas), 2);

for k=1:length(Fas)
	beam.Fex1(end-1) = Fas(k);  % Forcing from last node
	H1tmp = ((Kst-Ws^2*beam.M)+1j*(Ws*beam.D))\beam.Fex1;
	X0 = zeros(Nd*Nhc, 1); X0(Nd+(1:2*Nd)) = [real(H1tmp); -imag(H1tmp)];

	Sopt = struct('jac','full','stepmax',10000,'MaxFfunEvals',500, ...
        'dsmax', dsmax, 'Dscale', Dscale, 'dynamicDscale', 1);
	Xcont{k} = solve_and_continue(X0, ...
        @(X) HB_residual(X, beam, Nh, Nt, 'frf'), Ws, We, ds, Sopt);
	Sols{k}  = [Xcont{k}(end, :); 
        sqrt([1 0.5*ones(1,2*Nh)]*Xcont{k}(fdof:Nd:end-1,:).^2);
        atan2d(-Xcont{k}(2*Nd+fdof,:), Xcont{k}(Nd+fdof,:))]';
    [~,is] = unique(Sols{k}(:,3));
    Pks(k, :) = interp1(Sols{k}(is,3), Sols{k}(is,1:2), -90, 'pchip');
end

%% PNLSS
Alevel = '05';
load(sprintf('./TRANSIENT/famp%s/CLCLEF_MULTISINE.mat',Alevel), 'fsamp')
Alevel = 'comb';
load(sprintf('./pnlss%s.mat', Alevel),'model');

Ndpnlss = size(model.A,1);

% Forcing vector
Uc = zeros(Nh+1, 1);
Uc(2) = 1;

Wstart = Ws;
Wend = We;

Xpnlss = cell(size(Fas));
Solspnlss = cell(size(Fas));
for iex=1:length(Fas)
    Xc = (exp(1i*Wstart/fsamp)*eye(size(model.A))-model.A)\(model.B*Fas(iex));
    
    X0 = [zeros(length(model.A),1);real(Xc);-imag(Xc);....
            zeros(2*(Nh-1)*length(model.A),1)];                  % initial guess
%     Dscale = [mean(abs(Xc))*ones(length(X0),1);Wstart];

%     TYPICAL_x = Fas(iex)/(2*Zetas(imod)*Dst(imod)^2);    
%     Dscale = [TYPICAL_x*ones(length(X0),1);0.5*(Wstart+Wend)];
    
    TYPICAL_x = Fas(iex)/(2*Zetas(imod)*Dst(imod)^2);
    TYPICAL_xd = 0.5*(Wstart+Wend)*TYPICAL_x;
    TYPICAL_z = TYPICAL_x;
    Dscale = [repmat([TYPICAL_x; TYPICAL_xd; TYPICAL_z], 2*Nh+1, 1); (Wstart+Wend)/2];
    ds = 100;
    Sopt = struct('ds',ds,'dsmin',ds/5,'dsmax',ds*5,'flag',1,'stepadapt',1, ...
            'predictor','tangent','parametrization','arc_length', ...
            'Dscale',Dscale,'jac','full', 'dynamicDscale', 1,...
            'stepmax', 10000);
    
    fun_residual = ...
            @(XX) mhbm_aft_residual_pnlss_discrete(XX, model.A, ...
            model.B, model.E, model.xpowers, 1/fsamp, Uc*Fas(iex), Nh, Nt);
    Cfun_postprocess = {@(varargin) ...
            mhbm_post_amplitude_pnlss(varargin{:}, Uc*Fas(iex), model.C,...
            model.D, zeros(1,length(model.E)), model.xpowers, Nh,Nt)};
    fun_postprocess = @(Y) mhbm_postprocess(Y, fun_residual, ...
        Cfun_postprocess);

    [Xpnlss{iex},~,Sol] = solve_and_continue(X0, fun_residual,...
        Wstart, Wend, ds, Sopt, fun_postprocess);
    Solspnlss{iex} = [Xpnlss{iex}(end,:)' [Sol.Apv]' [Sol.Aph1]'];
end

%% Plot
fg1 = 10;
fg2 = 20;

figure(fg1)
clf()

figure(fg2)
clf()
colos = distinguishable_colors(length(Fas));
aa = gobjects(size(Fas));
for iex=1:length(Fas)
    figure(fg1)
    plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2)/Fas(iex), '-', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2)/Fas(iex), '.--', 'Color', colos(iex,:))
    
    figure(fg2)
    aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3), '-', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3), '.--', 'Color', colos(iex,:))
    legend(aa(iex), sprintf('F = %.2f', Fas(iex)));
end

figure(fg1)
xlim(sort([Ws We])/2/pi)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('RMS response amplitude (m)')
savefig(sprintf('./FIGURES/pnlssfrf_A%s_Amp.fig',Alevel))
print(sprintf('./FIGURES/pnlssfrf_A%s_Amp.eps',Alevel), '-depsc')

figure(fg2)
xlim(sort([Ws We])/2/pi)
ylim([-180 0])
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
legend(aa(1:end), 'Location', 'northeast')
savefig(sprintf('./FIGURES/pnlssfrf_A%s_Phase.fig',Alevel))
print(sprintf('./FIGURES/pnlssfrf_A%s_Phase.eps',Alevel), '-depsc')
