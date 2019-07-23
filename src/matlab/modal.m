function sd = modal(A,C)
% Calculate natural freqs and normalized modes from state space matrices
% A and C. Freqs are in Hz.
%
% USAGE:
%  sd = modal(A,C)
% INPUTS:
%  State space matrices in continous form. Convert from discrete form by:
%  sys = d2c(ss(A,B,C,D,1/fs));
%  sd = modal(sys.A,sys,C)
% OUTPUTS:
%  struct with fields: wn, wd, zeta, realmode, cpxmode
%
% https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
% ie.
% Ac = logm(Ad)*fs.
% Bc = Ac @ solve(Ad - np.eye(len(Ad)), Bd)
% and Cd = Cc and Dd = Dc

[egvec, egval] = eig(A);

% throw away very small values. Note this only works for state-space
% systems including damping. For undamped system, imag(lda) == 0!
% Use abs(imag(..)) if you want to retain complex conj. eigenvalues
egval = diag(egval);
idx = abs(imag(egval)) > 1e-8;
egval = egval(idx);
egvec = egvec(:,idx);

% sort after eigenvalues
[~, idx] = sort(imag(egval));
egval = egval(idx);
egvec = egvec(:,idx);
wd = imag(egval) / (2*pi);
wn = abs(egval) / (2*pi);

% Definition: sqrt(1 - (freq/natfreq)^2)
zeta = - real(egval) ./ abs(egval);

% cannot calculate mode shapes if C is not given
if nargin < 2 || norm(C) == 0
    realmode = egvec;
    cpxmode = egvec;
else
    % Transpose so cpxmode has format: (modes, nodes)
    cpxmode = (C * egvec).';
    realmode = real(cpxmode);
end

% normalize realmode
nmodes = size(realmode,1);
for i = 1:nmodes
    realmode(i,:) = realmode(i,:) / norm(realmode(i,:));
    if realmode(i,1) < 0
        realmode(i,:) = - realmode(i,:);
    end
end

sd.wn = wn;
sd.wd = wd;
sd.zeta = zeta;
sd.realmode = realmode;
sd.cpxmode = cpxmode;

end