%% Define beam subsystem
function [L,rho,E,ommod,PHI,dPHI,gam] = beams_for_everyone(setup,Nmod,thickness_input)

% Essential beam properties
load([setup '_Setup.mat'])
thickness=thickness_input;

% Auxiliary parameters
area = width*thickness;
rhoA = rho*area;
I = width*thickness^3/12;
EI = E*I;
r = thickness/2;

% Compute first normal modes (beam only)
W = @(k,x) [sin(k*x) cos(k*x) sinh(k*x) cosh(k*x)];
dW = @(k,x) [cos(k*x) -sin(k*x) cosh(k*x) sinh(k*x)];
ddW = @(k,x) [-sin(k*x) -cos(k*x) sinh(k*x) cosh(k*x)];
Wt = @(k,x) [sin(k*x);cos(k*x);sinh(k*x);cosh(k*x)];

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
    C(:,imod) = C(:,imod)/sqrt(mst);
    phi = @(x) ...
        C(1:4,imod)'*Wt(kmod(imod),x);
    
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

% % Plot modes
% x_lsp=linspace(0,L,100);
% for ii=1:100
%     x=x_lsp(ii);
%     PHI_x(ii,:)=PHI(x);
% end
% figure; plot(x_lsp/L,PHI_x); xlim([0,1])




