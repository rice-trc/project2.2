function [freqs, xf] = FFTFUN(t, x)
%FFTFUN implements fft on given signa and provides interpretable
%output
% 
% USAGE:
% 	[freqs, xf] = FFTFUN(t, x);
% INPUTS:
% 	t 	: Ntx1 time vector
% 	x 	: Ntxn matrix of data
% OUTPUTS:
%	freqs 	: (Nt/2)x1 vector of frequency values
%	xf	: (Nt/2)xn matrix of frequency components
    Nt = length(t);
    
    freqs = (0:((Nt-mod(Nt,2))/2-(1-mod(Nt,2))))/t(end);
    xf = fft(x)/(Nt/2);  xf(1,:) = xf(1,:)/2;
    xf = xf(1:(Nt/2),:);
end