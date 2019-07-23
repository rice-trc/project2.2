function [fex ,ms] = multisine_multi(f1, f2, N, A, Nt, ms_type, seed)
% Return the excited lines along with harmonics
% Used for time-domain multisine excitation, ie.
%
% phase = 2*pi*rand(N,1);
% fex = @(t) har'*A*cos(2*pi*(1:N)'*f0*t(:)' + phase) / sqrt(sum(har));
% 
if nargin == 7
    rng(seed)
end

f0 = (f2(1)-f1(1))/N;
lines = [];
for i=1:length(f1)
    linesMin = ceil(f1(i)/f0)+1;
    linesMax = floor(f2(i)/f0)+1;
    lines = [lines linesMin:linesMax];
end
% lines = linesMin:linesMax;

har = zeros(linesMax,1);
har(lines) = 1;

% convert lines to 1-index for using with fft-output.
lines = lines + 1;
% remove DC line
lines(lines==1) = [];

phase = 2*pi*rand(length(lines),1);
fex = @(t) A*sum(cos(2*pi*f0*lines'.*t(:)' + phase))/sqrt(length(lines)/2);

ms.phase = phase;
ms.lines = lines;
ms.har = har;
ms.f0 = f0;

end

