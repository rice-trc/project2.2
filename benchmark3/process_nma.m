function nma = process_nma(nma, Ntd)

% dimensionless time - exclude last point, ie. tau = [0, 2pi[
tau = linspace(0,2*pi,Ntd+1); tau = tau(1:end-1);

PHI_L2 = nma.PHI_L2;
n = nma.ndof;
Psi = nma.X(1:end-3,:);
Om = nma.X(end-2,:);
del_HB = nma.X(end-1,:);  % damping
log10a = nma.X(end,:);
a = 10.^log10a;
Q_HB = Psi.*repmat(a,size(Psi,1),1);

w_L2 = zeros(Ntd,length(Om)); 
a_q = zeros(n,length(Om));
Qrms = zeros(n,length(Om));

% loop over all modes
for k = 1:n
    % For peak-peak amplitude we need the time-periodic solution
    % get complex coefficients
    Qc = [Q_HB(k,:);Q_HB(n+k:2*n:end,:)-1i*Q_HB(2*n+k:2*n:end,:)];
    % time-periodic solution
    q = real(exp(1i*tau(:)*(0:nma.Nh))*Qc);
    w_mode = PHI_L2(k)*q;
    w_L2 = w_L2 + w_mode;
    a_q(k,:) = (max((q))-min((q)))/2;
    
    % For RMS value, we use parsevals relation:
    Qrms(k,:) = sqrt(sum(Q_HB(k:n:end,:).^2))/sqrt(2);
end

nma.arms = PHI_L2*Qrms;
nma.apeak = (max((w_L2))-min((w_L2)))/2;
nma.Qc = Qc;
nma.tau = tau;
nma.Ntd = Ntd;
nma.Om = Om;
nma.Qsc = Q_HB;
nma.a_q = a_q;
nma.Qrms = Qrms;
nma.xi = del_HB;
nma.Psi = Psi;
nma.a = a;
end