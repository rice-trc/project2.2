function Sol = mhbm_post_amplitude_pnlss(Sol,XcIN,~,UcIN,Ct,Dt,Ft,pp,Nh,Ntd)

    % Apply IDFT to every coordinate in x
    nx = size(Ct,2);
    Xc = transpose(reshape(XcIN,nx,Nh+1));
    Xc = [Xc(1,:); Xc(2:end,:)/2; zeros(Ntd-Nh-1,nx)];
    Xc(end-Nh+1:end,:) = flipud(conj(Xc(2:Nh+1,:)));
    x_tau = ifft(Xc)*Ntd;

    % Apply IDFT to u
    Uc = [UcIN(1); UcIN(2:end)/2; zeros(Ntd-Nh-1,1)];
    Uc(end-Nh+1:end,:) = flipud(conj(Uc(2:Nh+1,:)));
    u_tau = ifft(Uc)*Ntd;
    %% Evaluate nonlinear terms and apply DFT
    ny = size(Ct,1); nz = size(Ft,2);
    z_tau = prod(kron([x_tau u_tau],ones(size(pp,1),1)).^repmat(pp,Ntd,1),2);
	ynl_tau = (Ft*reshape(z_tau,nz,Ntd))';

    % Because of nonlinearities in the output, we need to consider more 
    % harmonics here
    Nhev = max(5*Nh,floor(Ntd/2)-2);

    Xtmp = fft(ynl_tau(end-Ntd+1:end,:))/Ntd;
    Xtmp = [Xtmp(1,:);Xtmp(2:Nhev+1,:)*2];
    Ynlc = reshape(transpose(Xtmp),[],1);

    % Setup linear part and add reshaped nonlinear part
    Yc = Ynlc;
    Yc(1:ny*(Nh+1)) = Yc(1:ny*(Nh+1)) + ...
        kron(eye(Nh+1),Ct)*XcIN + kron(eye(Nh+1),Dt)*UcIN;
    Sol.Yc = Yc;

    % Calculate amplitude
    Sol.A = zeros(ny,1); Sol.Amax = Sol.A;
    for i=1:ny
        Yi = [real(Yc(i));zeros(2*Nhev,1)];
        Yi(2:2:end) = real(Yc(ny+i:ny:end));
        Yi(3:2:end) = -imag(Yc(ny+i:ny:end));
        Sol.A(i) = mhbm_maximum(Yi) - Yi(1);
        Sol.Amax(i) = mhbm_maximum(Yi);

        Sol.Apv(i) = sqrt([1 0.5*ones(1, 2*Nhev)]*Yi.^2);  % Parseval's amplitude
        Sol.Aph1(i) = atan2d(-Yi(3),Yi(2));  % First harmonic phase
%         if Sol.Aph1(i)>0
%             Sol.Aph1(i) = 180-Sol.Aph1(i);
%         end
%         Sol.Aph1(i) = atan2d(-abs(Yi(3)),abs(Yi(2)));  % First harmonic phase
    end
end