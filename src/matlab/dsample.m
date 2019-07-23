function [Y,u] = dsample(y,dfactor,u)
% Low-pass filtering and downsampling.
%
% The input can be downsampled(ie. basically u(1:dfactor:end)) as it
% does mot contains higher harmonics, but y needs to be low-pass filtered
% in order to avoid aliasing.
%
% INPUT:
%  dfactor: downsample factor
% RETURN:
%  Y: decimated, u: downsampled

if nargin > 2
    [Nt,P,R,m] = size(u);
    u = reshape(u, [Nt*P,R,m]);
    u = downsample(u,dfactor);
end

[Nt,P,R,p] = size(y);
y = reshape(y, [Nt*P,R,p]);

% prime factor decomposition.
drate = factor(dfactor);
% length of decimated signal. Found from help of decimate
x = Nt*P;
for i = 1:length(drate)
    x = ceil(x/drate(i));
end
% decimated time points per period. Shall be integer!
N = x/P;
if rem(N,1) ~= 0
    error('The dfactor does not match number of periods and time points!')
end

Y = zeros(x,R,p);
for i=1:p
    for r=1:R
        ytmp = y(:,r,i);
        for k=1:length(drate)
            ytmp = decimate(ytmp,drate(k),'fir');
        end
        Y(:,r,i) = ytmp;
    end
end

% Removal of the last simulated period to eliminate the edge effects
% due to the low-pass filter.
Y = Y(1:(P-1)*N,:,:);
Y = reshape(Y, [N,P-1,R,p]);

if nargin > 2
    u = u(1:(P-1)*N,:,:);
    u = reshape(u, [N,P-1,R,m]);
end
if nargout == 2 && nargin ~= 3
    u = [];
end

end