%% mhbm_aft_residual_pnlss.m
function [R,dR_dXOm,dR_dX,dR_dOm,Fnlc,Xc,Om] = ...
    mhbm_aft_residual_pnlss_discrete(XO,At,Bt,Et,pp,dt,Uc,Nh,Ntd)

    X = XO(1:end-1);
    Om = XO(end);
    %% Real-to-complex conversion
    nx = size(At,2);
    i0 = 1:nx;
    icos = repmat(1:nx,1,Nh) + 2*nx*kron(1:Nh,ones(1,nx)) - nx;
    isin = icos+nx;
    Xc = [X(i0); X(icos)-1i*X(isin)];
    EYE = eye(length(X));
    dXc_dX = [EYE(i0,:);EYE(icos,:)-1i*EYE(isin,:)];
    dXc_dOm = zeros(size(Xc,1),1);
    dXc = [dXc_dX dXc_dOm];
    %% Calculation of the nonlinear forces and the Jacobian
    [Fnlc,dFnlc] = mhbm_pnlss_aft(Xc,dXc,pp,Et,Uc,Nh,Ntd);
    dFnlc_dX = dFnlc(:,1:end-1); dFnlc_dOm = dFnlc(:,end);
    %% Assembly of the residual and the Jacobian

    % Frequency domain derivative matrix and its derivative w.r.t. Om
    % D = 1i*Om*kron(diag(0:Nh),eye(nx));
    D = kron(diag(exp(1i*Om*dt*(0:Nh))),eye(nx));
    dd = 1i*dt*kron(diag(0:Nh),eye(nx));
    dD_dOm = D.*dd;

    % Residual in complex arithmetic
    A = kron(eye(Nh+1),At)-D; B = kron(eye(Nh+1),Bt);
    Rc = A*Xc+B*Uc+Fnlc;
    dRc_dX = A*dXc_dX+dFnlc_dX;
    dRc_dOm = -dD_dOm*Xc+dFnlc_dOm;
    %% Complex-to-real conversion
    R = zeros(size(X));
    R(i0) = real(Rc(1:nx));
    R(icos) = real(Rc(nx+1:end));
    R(isin) = -imag(Rc(nx+1:end));
    dR_dX = zeros(length(X));
    dR_dX(i0,:) = real(dRc_dX(1:nx,:));
    dR_dX(icos,:) = real(dRc_dX(nx+1:end,:));
    dR_dX(isin,:) = -imag(dRc_dX(nx+1:end,:));
    dR_dOm = zeros(size(X));
    dR_dOm(i0) = real(dRc_dOm(1:nx));
    dR_dOm(icos) = real(dRc_dOm(nx+1:end));
    dR_dOm(isin) = -imag(dRc_dOm(nx+1:end));

    dR_dXOm = [dR_dX dR_dOm];
end