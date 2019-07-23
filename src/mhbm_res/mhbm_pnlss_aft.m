function [FnlcOUT,dFnlcOUT] = mhbm_pnlss_aft(XcIN,dXcIN,pp,Et,Uc,Nh,Ntd)

%     Et = Et*dt*fs;

    %% Inverse DFT

    % Apply IDFT to every coordinate in x
    nx = size(Et,1);
    Xc = transpose(reshape(XcIN,nx,Nh+1));
    Xc = [Xc(1,:); Xc(2:end,:)/2; zeros(Ntd-Nh-1,nx)];
    Xc(end-Nh+1:end,:) = flipud(conj(Xc(2:Nh+1,:)));
    x_tau = ifft(Xc)*Ntd;  

    % Apply IDFT to u
    Uc = [Uc(1); Uc(2:end)/2; zeros(Ntd-Nh-1,1)];
    Uc(end-Nh+1:end,:) = flipud(conj(Uc(2:Nh+1,:)));
    u_tau = ifft(Uc)*Ntd;  % Sampled at Omega/(2pi)/Ntd
    
    %% Evaluate nonlinear terms and apply DFT
    z_tau = prod(kron([x_tau u_tau],ones(size(pp,1),1)).^repmat(pp,Ntd,1),2);
    nz = size(Et,2);
    f_tau = (Et*reshape(z_tau,nz,Ntd))';
    Xtmp = fft(f_tau(end-Ntd+1:end,:))/Ntd;
    Xtmp = [Xtmp(1,:);Xtmp(2:Nh+1,:)*2];
    FnlcOUT = reshape(transpose(Xtmp),[],1);
    
    %% Evaluate gradients of nonlinear terms and apply DFT
    dFnlcOUT = 0*dXcIN;
    for l=1:nx
        % Apply IDFT to Jacobian of x
        ndx = size(dXcIN,2);
        dXc = [dXcIN(l,:); dXcIN(nx+l:nx:end,:)/2; zeros(Ntd-Nh-1,ndx)];
        dXc(end-Nh+1:end,:) = flipud(conj(dXc(2:Nh+1,:)));
        dxl_dtau = ifft(dXc)*Ntd;

        notl = setdiff(1:nx,l);
        dzxl_dxl = repmat(pp(:,l),Ntd,1).*...
            (kron(x_tau(:,l),ones(size(pp,1),1)).^repmat(pp(:,l)-1,Ntd,1));
        dz_dxl = prod([ dzxl_dxl ...
            kron([x_tau(:,notl) u_tau],ones(size(pp,1),1)).^...
            repmat(pp(:,[notl end]),Ntd,1)], 2);
        df_dxl = (Et*reshape(dz_dxl,nz,Ntd))';

        for i=1:nx
            dfi_dX = repmat(df_dxl(:,i),1,size(dxl_dtau,2)).*dxl_dtau;
            Xtmp = fft(dfi_dX(end-Ntd+1:end,:))/Ntd;
            Xtmp = [Xtmp(1,:);Xtmp(2:Nh+1,:)*2];
            dFnlcOUT(i:nx:end,:) = dFnlcOUT(i:nx:end,:) + Xtmp;
        end
    end
end