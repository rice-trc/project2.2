function frf=load_hb(fname, Ntd, PHI)

% time points in period
% Ntd = 1e3;
% dimensionless time - exclude last point, ie. tau = [0, 2pi[
tau = linspace(0,2*pi,Ntd+1); tau = tau(1:end-1);

% load data and process
frf = load(fname);
n = frf.n;
% different ext. levels
nex = length(frf.exc_lev);

a_rms = cell(1,nex);
a_diff = cell(1,nex);
Om = cell(1,nex);
for j = 1:nex
    % Interpret solver output
    Om{j} = frf.X{j}(end,:);
    Q_HB = frf.X{j}(1:end-1,:);
    
    % two ways to define amplitude (at x = L/2)
    % 1) difference between min/max amplitude in time domain or
    % 2) RMS, as found by persevals identity, eq. 2.18 p 21 in (Malte book)
    % https://sci-hub.tw/10.1007/978-3-030-14023-6
    
    % w_L2: [Time points, cont. step]
    w_L2 = zeros(Ntd,length(Om{j}));
    a_rms_loc = 0;
    % loop over all modes(or DOF in the discrete case)
    for k = 1:n
        % 1) convert Q_sc -> Q_ce for finding the periodic sol in time
        % include all harmonics
        Qc = [Q_HB(k,:);Q_HB(n+k:2*n:end,:)-1i*Q_HB(2*n+k:2*n:end,:)];
        % get displacement at center caused by each mode in time domain
        w_mode = PHI(k)*real(exp(1i*tau(:)*(0:frf.H))*Qc);
        % sum up to actual displacement at center in time domain
        w_L2 = w_L2 + w_mode;

%         q = real(exp(1i*tau(:)*(0:frf.H))*Qc);
%         a_q(k,:) = (max((q))-min((q)))/2;

        % 2) RMS. include all harmonics
        Q_rms = sqrt(sum(Q_HB(k:n:end,:).^2))/sqrt(2);
        a_rms_loc = a_rms_loc + PHI(k)*Q_rms;
    end
    % compute peak to peak amplitude
    a_diff{j} = (max((w_L2))-min((w_L2)))/2;     
    a_rms{j} = a_rms_loc;
end

frf.Om = Om;
frf.Qsc = Q_HB;
frf.Qc = Qc;
frf.apeak = a_diff;
frf.arms = a_rms;
frf.nex= nex;
frf.tau = tau;

end