%% mhbm_maximum.m
% Calculate maximum time domain value of given harmonic polynomial
function [wmax,taumax,Htaumax] = mhbm_maximum(w)
%% Compute auxiliary variables
epsrel = 1e-10; epsabs = 1e-12;
Nh = (length(w)-1)/2;

% Differentiation matrix
D = zeros(2*Nh,1);
D(2:2:end) = 1:Nh;
D = diag(D,1)-diag(D,-1);
%% Compute maxima of truncated fourier series
tau0 = 0;
ikeydir = -1;
[~,~,taumax] = mhbm_rootfinding(D*w,tau0,ikeydir,epsrel,epsabs);
wmax = zeros(size(taumax));
Htaumax = cell(size(taumax));
for ii=1:length(taumax)
    Htaumax{ii} = [1, zeros(1,2*Nh)];
    Htaumax{ii}(2:2:end-1) = cos((1:Nh)*taumax(ii));
    Htaumax{ii}(3:2:end) = sin((1:Nh)*taumax(ii));
    wmax(ii) = Htaumax{ii}*w;
end
%% Select only absolute maximum
[wmax, indmax] = max(wmax);
taumax = taumax(indmax);
Htaumax = Htaumax{indmax};