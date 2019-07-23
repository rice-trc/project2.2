clc
clear all

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
imod = 3;  % Desired mode

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
end

%% Harmonic Convergence based on FRF
Hs = [1:2:40 40];
if isfile(sprintf('HConvDat_M%d.mat',imod))
    load(sprintf('HConvDat_M%d.mat',imod), 'PkPs', 'Errs', 'errmax', 'Nhconv', 'ihconv');
else
    PkPs = zeros(length(Hs), 2);
    Sc = zeros(1, 3);

    for ih=1:length(Hs)
        Nh = Hs(ih);
        Nhc = 2*Nh+1;

        beam.Fex1(end-1) = Fas(3);  % Forcing from last node
        H1tmp = ((Kst-Wsc^2*beam.M)+1j*(Wsc*beam.D))\beam.Fex1;
        X0 = zeros(Nd*Nhc, 1); X0(Nd+(1:2*Nd)) = [real(H1tmp); -imag(H1tmp)];

        Dscale = [1e-4*ones(Nd*Nhc,1); (Dst(1)+Dsl(1))/2];
        Sopt = struct('jac','full','stepmax',1000,'MaxFfunEvals',500, ...
            'dsmax', dsmax, 'Dscale', Dscale, 'dynamicDscale', 1);

        Xc = solve_and_continue(X0, ...
            @(X) HB_residual(X, beam, Nh, Nt, 'FRF'), Wsc, Wec, ds, Sopt);
        Sc = [Xc(end, :); 
            sqrt([1 0.5*ones(1,2*Nh)]*Xc(fdof:Nd:end-1,:).^2);
            atan2d(-Xc(2*Nd+fdof,:), Xc(Nd+fdof,:))]';
        PkPs(ih,:) = interp1(Sc(:,3), Sc(:,1:2), -90, 'pchip');

        fprintf('%d/%d Done.\n', ih, length(Hs))     
    end

    % Convergence Criterion
    errmax = 0.01*1e-2;
    Errs = abs(PkPs-PkPs(end,:))./PkPs;
    ihconv = find(max(Errs,[],2)<=errmax, 1 );
    Nhconv = Hs(ihconv);

    save(sprintf('HConvDat_M%d.mat',imod), 'PkPs', 'Errs', 'errmax', 'Nhconv', 'ihconv')
end
%% Plotting Harmonic Convergence
figure(100)
clf()
semilogy(Hs(1:end-2), Errs(1:end-2,:)*100, '.-', 'LineWidth', 2); hold on
semilogy([0 Hs], ones(size([0 Hs]))*errmax*100, 'k--', 'LineWidth', 2)
semilogy(Hs(ihconv), Errs(ihconv,:)*100, 'ko', 'MarkerFaceColor', 'k')
xlabel('Number of Harmonics')
ylabel('Relative Error (\%)')
legend('Resonant frequency', 'Resonant amplitude', sprintf('Threshold: %.2f \\%%', errmax*100), sprintf('Convergence: $N_h$=%d',Nhconv))
% print(sprintf('./FIGURES/HConv_M%d.eps',imod), '-depsc')

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

%% NMA
Nh = Nhconv;
Nhc = 2*Nh+1;
Dscale = [1e-1*ones(Nd*Nhc,1); Dst(imod); Zetas(imod); 1.0];

inorm = Nd-1;

X0 = zeros(Nd*Nhc+2, 1);
X0(Nd+(1:Nd)) = Vst(:,imod);
X0(end-1) = Dst(imod);
X0(end) = Zetas(imod);

beam.Fex1 = beam.Fex1*0;

Sopt = struct('jac', 'full', 'Dscale', Dscale, 'dynamicDscale', 1, ...
    'dsmax', dl10amax);

fscl = mean(abs(beam.K*Vst(:,imod)));
Xbb = solve_and_continue(X0, ...
    @(X) HB_residual(X, beam, Nh, Nt, 'nma', inorm, fscl), ...
    log10a_s, log10a_e, dl10a, Sopt);
Bkb = [10.^Xbb(end,:);  % modal amplitude
    Xbb(end-2,:);  % frequency
    Xbb(end-1,:);  % damping factor
    atan2d(-Xbb(2*Nd+fdof,:), Xbb(Nd+fdof,:)); % phase
    (10.^Xbb(end,:)).*sqrt([1 0.5*ones(1, 2*Nh)]*Xbb(fdof:Nd:end-3,:).^2)]';

%% Plotting FRF
colos = distinguishable_colors(length(Fas), 'k');

% NM-ROM Parameters
zts = Bkb(:,3);
oms = Bkb(:,2);
p2  = oms.^2-2*(oms.*zts).^2;
om4 = oms.^4;
Phi_HB = Xbb(Nd+(1:Nd),:)-1j*Xbb(2*Nd+(1:Nd),:);
Fsc = (abs(Phi_HB'*Fex1)./Bkb(:,1)).^2;
mAmps = Bkb(:,5);
php = rad2deg(angle(Phi_HB'*Fex1));
phf = Phi_HB'*Fex1;

figure(1)
clf()
semilogy(Bkb(:, 2)/(2*pi), Bkb(:, end), 'k--', 'LineWidth', 2); hold on

figure(2)
clf()
livs = 10;
aa = gobjects(size(Fas));
for k=1:length(Fas)
    om1 = sqrt(p2 + sqrt(p2.^2-om4 + Fsc*Fas(k)^2));
    om2 = sqrt(p2 - sqrt(p2.^2-om4 + Fsc*Fas(k)^2));
    
    ris1 = find(imag(om1)==0);
    ris2 = find(imag(om2)==0);
    
% 	phi1 = php(ris1)+atan2d(-2*zts(ris1).*oms(ris1).*om1(ris1), oms(ris1).^2-om1(ris1).^2);
%     phi2 = php(ris2)+atan2d(-2*zts(ris2).*oms(ris2).*om2(ris2), oms(ris2).^2-om2(ris2).^2);

% Have to be offsetted by -180 degrees for mode 1
    phi1 = rad2deg(angle(phf(ris1)./((oms(ris1).^2-om1(ris1).^2)+1j*(2*zts(ris1).*oms(ris1).*om1(ris1)))));
    phi2 = rad2deg(angle(phf(ris2)./((oms(ris2).^2-om2(ris2).^2)+1j*(2*zts(ris2).*oms(ris2).*om2(ris2)))));
    
    figure(1)
    semilogy(Sols{k}(:,1)/2/pi, Sols{k}(:,2), '-', 'LineWidth', 2, 'Color', colos(k,:)); hold on
    semilogy(om1(ris1)/2/pi, mAmps(ris1), '--', 'Color', colos(k,:))
    semilogy(om2(ris2)/2/pi, mAmps(ris2), '--', 'Color', colos(k,:))
	ivs = fix(linspace(1, length(ris1), livs));
    semilogy(om1(ris1(ivs))/2/pi, mAmps(ris1(ivs)), '.', 'Color', colos(k,:), 'MarkerSize', 20)
    ivs = fix(linspace(1, length(ris2), livs));
    semilogy(om2(ris2(ivs))/2/pi, mAmps(ris2(ivs)), '.', 'Color', colos(k,:), 'MarkerSize', 20)
	if k==3
        semilogy(Pks(k,1)/2/pi, Pks(k,2), 'k.', 'MarkerSize', 30)
    end
    
    figure(2)
    aa(k) = plot(Sols{k}(:,1)/2/pi, Sols{k}(:,3), '-', 'LineWidth', 2, 'Color', colos(k,:)); hold on
    legend(aa(k), sprintf('Fa = %.2f', Fas(k)))
    plot(om1(ris1)/2/pi, phi1, '--', 'Color', colos(k,:))
    plot(om2(ris2)/2/pi, phi2, '--', 'Color', colos(k,:))
    ivs = fix(linspace(1, length(ris1), livs));
    plot(om1(ris1(ivs))/2/pi, phi1(ivs), '.', 'Color', colos(k,:), 'MarkerSize', 20)
    ivs = fix(linspace(1, length(ris2), livs));
    plot(om2(ris2(ivs))/2/pi, phi2(ivs), '.', 'Color', colos(k,:), 'MarkerSize', 20)
	if k==3
        plot(Pks(k,1)/2/pi, -90, 'k.', 'MarkerSize', 30)
    end
end
figure(1)
xlabel('Forcing Frequency $\omega$ (Hz)')
ylabel('RMS Response Displacement Amplitude (m)')
xlim(xls)
ylim(yls)
print(sprintf('./FIGURES/FResp_M%d.eps',imod), '-depsc')

figure(2)
legend(aa(1:end), 'Location', 'northeast')
xlabel('Forcing Frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
xlim(xls)
print(sprintf('./FIGURES/PhResp_M%d.eps',imod), '-depsc')

%% Plotting NM Backbone
figure(10)
clf()
semilogx(Bkb(:, 1), Bkb(:, 2)/2/pi, 'LineWidth', 2)
xlabel('Modal Amplitude')
ylabel('Natural Frequency (Hz)')
print(sprintf('./FIGURES/NMA_Freq_M%d.eps', imod), '-depsc')

figure(20)
clf()
loglog(Bkb(:, 1), Bkb(:, 3), 'LineWidth', 2)
xlabel('Modal Amplitude')
ylabel('Damping Factor')
print(sprintf('./FIGURES/NMA_Damp_M%d.eps', imod), '-depsc')