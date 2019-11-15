clc
clear all
addpath('../src/mhbm_res/')
addpath('../src/matlab/')
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')
addpath('../src/nlvib_hillsexp/')

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
kn  = 1.3e6;
gap = 1e-3;
add_nonlinear_attachment(beam, Nnl, dir, 'unilateralspring', ...
    'stiffness', kn, 'gap', gap);
Nd = size(beam.M, 1);

%% Linearized limit cases
% Slipped
[Vsl, Dsl] = eig(beam.K, beam.M);
[Dsl, si] = sort(sqrt(diag(Dsl)));
Vsl = Vsl(:, si);  Vsl = Vsl./sqrt(diag(Vsl'*beam.M*Vsl)');

% Stuck
Knl = zeros(size(beam.M));
nli = find(beam.nonlinear_elements{1}.force_direction);
Knl(nli, nli) = kn;
Kst = beam.K + Knl;
[Vst, Dst] = eig(Kst, beam.M);
[Dst, si] = sort(sqrt(diag(Dst)));
Vst = Vst(:, si); Vst = Vst./sqrt(diag(Vst'*beam.M*Vst));

%% Rayleigh damping
% Desired
zs = [8e-3; 8e-3];
ab = [ones(length(zs),1) Dst(1:length(zs)).^2]\(2*zs.*Dst(1:length(zs)));

beam.D = ab(1)*beam.M + ab(2)*Kst;

Zetas = diag(Vst'*beam.D*Vst)./(2*Dst)

Zetas_req = 8e-3*ones(beam.n,1);

beam.D = inv(Vst')*diag(2*Dst.*Zetas_req)*inv(Vst);

%% HCB Reduction
[Mhcb, Khcb, Thcb] = HCBREDUCE(beam.M, beam.K, (Nnl-2)*2+1, 3);
beam_hcb = struct('M', Mhcb, 'K', Khcb, 'L', beam.L*Thcb, 'n', size(Khcb,1), ...
    'D', Thcb'*beam.D*Thcb, 'Fex1', Thcb'*beam.Fex1, 'nonlinear_elements', {beam.nonlinear_elements});
beam_hcb.nonlinear_elements{1}.force_direction = Thcb'*beam.nonlinear_elements{1}.force_direction;
Nd_hcb = beam_hcb.n;
Fex1_hcb = Thcb'*Fex1;

Kst_hcb = Thcb'*Kst*Thcb;
%% HB properties
Nt = 2^10;
imod = 1;  % Desired mode

switch imod
    case 1
        Fas = [5.0 10.0 20.0 40.0 80.0 120.0 200.0];

        Ws = 200;
        We = 450;
        ds = abs(We-Ws)/100;
        dsmax = abs(We-Ws)/2;
        xls = [25 85];
        yls = [1e-5 1e-1];
        
        Wsc = 45*2*pi;
        Wec = 55*2*pi;
        
        log10a_s = -4;
        log10a_e = -0.5;
        dl10a = 0.01;
        dl10amax = 0.05;
    case 2
        Fas = [20.0 40.0 80.0 120.0 200.0 400.0];

        Ws = 225*2*pi;
        We = 380*2*pi;
        ds = abs(We-Ws)/100;
        dsmax = abs(We-Ws)/2;
        xls = [225 375];
        yls = [5e-6 5e-2];
        
        Wsc = 295*2*pi;
        Wec = 310*2*pi;
        
        log10a_s = -4.5;
        log10a_e = -0.5;
        dl10a = 0.01;
        dl10amax = 0.05;
    case 3
        Fas = [80.0 200.0 600.0 1200.0 2000.0];

        Ws = 820*2*pi;
        We = 870*2*pi;
        ds = abs(We-Ws)/100;
        dsmax = abs(We-Ws)/2; 
        xls = [800 900];
        yls = [1e-5 2e-2];
        
        Wsc = 840*2*pi;
        Wec = 850*2*pi;
        
        log10a_s = -4.5;
        log10a_e = -0.5;
        dl10a = 0.0001;
        dl10amax = 0.05;
end
load(sprintf('HConvDat_M%d.mat',imod), 'PkPs', 'Errs', 'errmax', 'Nhconv', 'ihconv')

%% FRF
Nh = Nhconv;
Nhc = 2*Nh+1;
Dscale = [1e-4*ones(Nd_hcb*Nhc,1); (Dst(1)+Dsl(1))/2];

Xcont = cell(size(Fas));
Sols = cell(size(Fas));
Pks = zeros(length(Fas), 2);

for k=1:length(Fas)
    beam_hcb.Fex1 = Fex1_hcb*Fas(k);  % Forcing from last node
	H1tmp = ((Kst_hcb-Ws^2*beam_hcb.M)+1j*(Ws*beam_hcb.D))\beam_hcb.Fex1;
	X0 = zeros(Nd_hcb*Nhc, 1); X0(Nd_hcb+(1:2*Nd_hcb)) = [real(H1tmp); -imag(H1tmp)];

	Sopt = struct('jac','full','stepmax',10000,'MaxFfunEvals',500, ...
        'dsmax', dsmax, 'Dscale', Dscale, 'dynamicDscale', 1);
	Xcont{k} = solve_and_continue(X0, ...
        @(X) HB_residual(X, beam_hcb, Nh, Nt, 'frf'), Ws, We, ds, Sopt);

    Sols{k} = [Xcont{k}(end, :);
        sqrt([1 0.5*ones(1,2*Nh)]*(kron(eye(Nhc), Thcb(fdof,:))*Xcont{k}(1:end-1,:)).^2);
        atan2d(-Thcb(fdof,:)*Xcont{k}(2*Nd_hcb+(1:Nd_hcb),:), Thcb(fdof,:)*Xcont{k}(Nd_hcb+(1:Nd_hcb),:))]';
    
    Pks(k, :) = interp1(Sols{k}(:,3), Sols{k}(:,1:2), -90, 'pchip');
end
beam.Fex1 = Fex1;
save('data/Fresp.mat', 'Sols', 'Fas', 'Ws', 'We', 'beam', 'beam_hcb', 'Fex1_hcb', 'Thcb')

%% Compute frequency response of PNLSS identified model
% N = 1e3;
exc_lev = Fas;
% Alevels = 0.5*N*exc_lev.^2;
% Alevels = [1,5,10,15,30,60,120];
Alevels = [15, 30, 60, 120];
nx = [2 3];
na = 2;

upsamp = 4;
dataname = 'ms_full';
load(sprintf('data/b%d_A%d_up%d_%s',5,Alevels(1),upsamp,dataname), 'fs')

Ws = 450;
We = 200;
% We = We-Ws;
% Ws = Ws+We;
% We = Ws-We;

for ia = 1:length(Alevels)
Alevel = Alevels(ia);
load(sprintf('data/pnlssmodel_b%d_shaker_A%d_up%d_%s_na%d_nx%s.mat',5,Alevels(ia),upsamp,dataname, na, sprintf('%d',nx)), 'model');
% load(sprintf('./data/pnlssout_A%d_F%d.mat',Alevel,fs),'model');

Ndpnlss = size(model.A,1);

% Forcing vector
Uc = zeros(Nh+1,1);
Uc(2) = 1;

ds = 50*2*pi;
dsmin = 0.0001*2*pi;
dsmax = 100*2*pi;

Xpnlss = cell(length(exc_lev),1);
Solspnlss = cell(length(exc_lev),1);
for iex=1:length(exc_lev)
    Ff = exc_lev(iex);

    Xc = (exp(1i*Ws/fs)*eye(size(model.A))-model.A)\(model.B*Ff);             % linear solution
    % Xc = ((1i*Om_s/fs)*eye(size(model.A))-model.A)\(model.B*Ff);             % linear solution
    X0 = [zeros(length(model.A),1);real(Xc);-imag(Xc);....
            zeros(2*(Nh-1)*length(model.A),1)];                  % initial guess
    
% 	TYPICAL_x = 1e5*Ff/(2*D*M*om^2);
    TYPICAL_x = 1e-4;
    Dscale = [TYPICAL_x*ones(length(X0),1);Ws];
    Sopt = struct('ds',ds,'dsmin',dsmin,'dsmax',dsmax,'flag',1,'stepadapt',1, ...
            'predictor','tangent','parametrization','arc_length', ...
            'Dscale',Dscale,'jac','full', 'dynamicDscale', 1, 'stepmax', 2000);

    fun_residual = ...
            @(XX) mhbm_aft_residual_pnlss_discrete(XX, model.A, model.B, model.E, model.xpowers, 1/fs, Uc*Ff, Nh, Nt);
    Cfun_postprocess = {@(varargin) ...
            mhbm_post_amplitude_pnlss(varargin{:},Uc*Ff,model.C,model.D,zeros(1,length(model.E)),model.xpowers,Nh,Nt)};
    fun_postprocess = @(Y) mhbm_postprocess(Y,fun_residual,Cfun_postprocess,Nh,model.n,fs);

    [Xpnlss{iex},~,Sol] = solve_and_continue(X0, fun_residual,...
        Ws, We, ds, Sopt, fun_postprocess);
    Solspnlss{iex} = [Xpnlss{iex}(end,:)' [Sol.Apv]' [Sol.Aph1]' [Sol.stab]' [Sol.unstab]'];
end
save(sprintf('./data/pnlssfresp_shaker_A%.2f_F%d_nx%s.mat',Alevel,fs,sprintf('%d',nx)), 'Solspnlss');

%% Plot
figure(10*ia)
clf()

figure(10*ia+1)
clf()
colos = distinguishable_colors(length(exc_lev));
aa = gobjects(size(exc_lev));
for iex=1:length(exc_lev)
    figure(10*ia)
    plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-', 'Color', colos(iex,:)); hold on
%     plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2).*Sols{iex}(:,4), '-', 'Color', colos(iex,:)); hold on
%     plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2).*Sols{iex}(:,5), '--', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2), '.-', 'Color', colos(iex,:)); hold on
%     plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2).*Solspnlss{iex}(:,4), '.-', 'Color', colos(iex,:)); hold on
%     plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2).*Solspnlss{iex}(:,5), '+-', 'Color', colos(iex,:)); hold on
    
    figure(10*ia+1)
    aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3), '-', 'Color', colos(iex,:)); hold on
%     aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3).*Sols{iex}(:,4), '-', 'Color', colos(iex,:)); hold on
%     plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3).*Sols{iex}(:,5), '--', 'Color', colos(iex,:)); hold on
    plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3), '.-', 'Color', colos(iex,:)); hold on
%     aa(iex) = plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3).*Solspnlss{iex}(:,4), '.-', 'Color', colos(iex,:)); hold on
%     plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3).*Solspnlss{iex}(:,5), '+-', 'Color', colos(iex,:)); hold on
    legend(aa(iex), sprintf('F = %.2f', exc_lev(iex)));
end

figure(10*ia)
set(gca, 'YScale', 'log')
xlim(sort([Ws We])/2/pi)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('RMS response amplitude (m)')
% savefig(sprintf('./fig/pnlssfrf_A%d_Amp.fig',Alevels(ia)))
% print(sprintf('FIGURES/pnlssfrf_Amp_b%d_A%d_up%d_%s_na%d_nx%s.eps',5,Alevels(ia),upsamp,dataname,na,sprintf('%d',nx)), '-depsc')
% print('./fig/stabsol_Amp.eps', '-depsc')
% print('./fig/dtstabsol_Amp.eps', '-depsc')

figure(10*ia+1)
xlim(sort([Ws We])/2/pi)
ylim([-180 180])
yticks(-180:45:180)
xlabel('Forcing frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
legend(aa(1:end), 'Location', 'northeast')
% savefig(sprintf('./fig/pnlssfrf_A%d_Phase.fig',Alevels(ia)))
% print(sprintf('./fig/pnlssfrf_A%.2f_Phase_nx%s.eps',Alevels(ia),sprintf('%d',nx)), '-depsc')
% print(sprintf('FIGURES/pnlssfrf_Phase_b%d_A%d_up%d_%s_na%d_nx%s.eps',5,Alevels(ia),upsamp,dataname,na,sprintf('%d',nx)), '-depsc')
% print('./fig/stabsol_Phase.eps', '-depsc')
% print('./fig/dtstabsol_Phase.eps', '-depsc')
end