clear; close all; clc;
%% Simulation of the self-adaptive system (beam with slider)

% Parameters to be varied
Nmod = 5;

%% Define beam subsystem

% define setup
setup = 'New_Design_Steel';

%% Define beam subsystem
load([setup '_Setup.mat'])
thickness = 0.0005;
% Auxiliary parameters
area = width*thickness;
rhoA = rho*area;
I = width*thickness^3/12;
EI = E*I;

% Compute first normal modes (beam only)

W = @(k,x) [sin(k*x) cos(k*x) sinh(k*x) cosh(k*x)];
Wt = @(k,x) [sin(k*x);cos(k*x);sinh(k*x);cosh(k*x)];

dW = @(k,x) [cos(k*x) -sin(k*x) cosh(k*x) sinh(k*x)];
dWt = @(k,x) [cos(k*x);-sin(k*x);cosh(k*x);sinh(k*x)];

ddW = @(k,x) [-sin(k*x) -cos(k*x) sinh(k*x) cosh(k*x)];
ddWt = @(k,x) [-sin(k*x);-cos(k*x);sinh(k*x);cosh(k*x)];

% Boundary condition matrix (clamped-clamped)
A = @(k) [...
    W(k,0); dW(k,0); W(k,L); dW(k,L)];
detA = @(k) det(A(k));

% Estimate eigenvalues
kest = (2*(1:Nmod)+1)*pi/2/L;

% Determine first modes
kmod = zeros(Nmod,1); C = zeros(4,Nmod);
ommod = zeros(Nmod,1); gam = zeros(Nmod,1);
for imod=1:Nmod
    kmod(imod) = fzero(@(x) detA(x),kest(imod));
    
    % Determine mode shape
    Phi = null(A(kmod(imod)));
    C(:,imod) = Phi;
    
    % Phase normalization
    inmlz = 1;
    C(:,imod) = C(:,imod)/C(inmlz,imod);
    
    % Normalize modes w.r.t. mass
    phi = @(x) ...
        C(1:4,imod)'*Wt(kmod(imod),x);
    
    mst = integral(@(x) rhoA*phi(x).^2,0,L);
    
    test = integral(@(x) rhoA*phi(x).*phi(x),0,L);
    
    C(:,imod) = C(:,imod)/sqrt(mst);
    phi = @(x) C(1:4,imod)'*Wt(kmod(imod),x);
    
    phi_save{imod} = @(x) C(1:4,imod)'*Wt(kmod(imod),x);
    dphi_save{imod} = @(x) C(1:4,imod)'*dWt(kmod(imod),x)*kmod(imod);
    ddphi_save{imod} = @(x) C(1:4,imod)'*ddWt(kmod(imod),x)*kmod(imod)^2;
    
    % Excitation coefficient
    gam(imod) = integral(@(x) rhoA*phi(x),0,L);
    
    % Natural frequency
    ommod(imod) = sqrt(EI*kmod(imod).^4/rhoA);
end

% Define continuous beam displacement as a function of x
phistr = '@(x,W,kmod,C) [';
dphistr = '@(x,dW,kmod,C) [';
ddphistr = '@(x,ddW,kmod,C) [';
for imod=1:Nmod
    phistr = [phistr ' W(x,kmod(' num2str(imod) ...
        '))*C(:,' num2str(imod) ') '];
    dphistr = [dphistr ' kmod(' num2str(imod) ...
        ')*dW(x,kmod(' num2str(imod) '))*C(:,' num2str(imod) ') '];
    ddphistr = [ddphistr ' kmod(' num2str(imod) ...
        ')^2*ddW(x,kmod(' num2str(imod) '))*C(:,' num2str(imod) ') '];
end
phistr = [phistr ']']; dphistr = [dphistr ']']; ddphistr = [ddphistr ']'];
PHItmp = str2func(phistr);
dPHItmp = str2func(dphistr);
ddPHItmp = str2func(ddphistr);
PHI = @(x) PHItmp(x,W,kmod,C);
dPHI = @(x) dPHItmp(x,dW,kmod,C);
ddPHI = @(x) ddPHItmp(x,ddW,kmod,C);


% Plot modes

x_lsp=linspace(0,L,500);
for ii=1:500
    x=x_lsp(ii);
    PHI_x(ii,:)=PHI(x);
end
figure; plot(x_lsp/L,PHI_x); xlim([0 x_lsp(end)/L]), grid
xlabel('X coordinate (m)')
ylabel('Y deflection magnitude')
% legend('1','2','3','4','5','6','7','8','9','10')

% mu = @(i,j,k,n) (-E/(2*L*rho)*quadgk(@(x) ...
%     dphi_save{i}(x).*dphi_save{j}(x),0,L,'AbsTol',1e-2) ...
%     *quadgk(@(x) ddphi_save{k}(x).*phi_save{n}(x),0,L,'AbsTol',1e-2)...
%     /quadgk(@(x) phi_save{n}(x).*phi_save{n}(x),0,L,'AbsTol',1e-2));

mu2 = @(i,j,k,n) (-E*width*thickness/(2*L)*quadgk(@(x) ...
    dphi_save{i}(x).*dphi_save{j}(x),0,L,'AbsTol',1e-2) ...
    *quadgk(@(x) ddphi_save{k}(x).*phi_save{n}(x),0,L,'AbsTol',1e-2));


rho*width*thickness*quadgk(@(x) phi_save{1}(x).*phi_save{1}(x),0,L,'AbsTol',1e-2) %must be 1!!!


%% calculate nonlinear coefficients, adding coefficients for equivalent
% term, so that only one term is needed later, for example:
% (p+q+r)*q_1^2*q_2 instead of p*q_1^2*q_2 + q*q_1*q_2*q_1 + r*q_2*q_1^2

b = zeros(Nmod,Nmod,Nmod,Nmod);
for n=1:Nmod
    for i=1:Nmod
        for j=1:Nmod
            for k=1:Nmod
                srtd_idx = sort([i j k]);
                b(srtd_idx(1),srtd_idx(2),srtd_idx(3),n) = b(srtd_idx(1),srtd_idx(2),srtd_idx(3),n) + mu2(i,j,k,n);
            end
        end
    end
end

mu2(1,1,1,1)


%% plot coeffs

counter = 1; bar_data = []; bar_data_ref = []; label_data = [];

% convert cubic coefficients to plot data
for rr = 1:Nmod
    for jj = 1:Nmod
        for kk = jj:Nmod
            for ll = kk:Nmod
                bar_data = [bar_data , sign(b(jj,kk,ll,rr))*log10(abs(b(jj,kk,ll,rr)))];
                label_data = [label_data ; sprintf('b_{%d,%d,%d}^{(%d)}',jj,kk,ll,rr)];
                counter = counter + 1;
            end
        end
    end
end

% % plot and label everything...
% figure('units','normalized','outerposition',[-1 0 1 1])
% 
% bar(1:counter-1,[bar_data']), hold on
% 
% ylabel('$\mathrm{sign}(b)\cdot \mathrm{log}_{10}(|b|)$')
% set(gca, 'XTick', 1:counter-1)
% set(gca,'xticklabel',label_data)
% grid
