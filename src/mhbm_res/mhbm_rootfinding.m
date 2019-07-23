function [tau, dudtau,tautmp] = ...
    mhbm_rootfinding(uh, tau0, ikeydir, epsrel, epsabs)
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%                                                                             ::
% Function 'func_myzerocrossing'                                              ::
%                                                                             ::
% Version:                                                                    ::
%   1.0, 01.12.2008, Matlab R14 SP1, Original code                            ::
%   1.1, 05.12.2008, Matlab R14 SP1, First modification                       ::
%   1.2, 10.12.2008, Matlab R14 SP1, Second modification                      ::
%                                                                             ::
% Programmer:                                                                 ::
%   Christian Siewert, Institute of Dynamics and Vibration Research           ::
%                                                                             ::
% Purpose:                                                                    ::
%   Function for the calculation of a particular zero crossing time instant   ::
%   of a truncated Fourier series. The method is based on the computation of  ::
%   the roots af the accompanying complex algebraic polynomial.               ::
%                                                                             ::
% Called functions:                                                           ::
%   --                                                                        ::
%                                                                             ::
% Input data dictionary:                                                      ::
%   uh                  Rmatrix         Vector of the harmonic coefficients   ::
%   tau0                Rscalar         Lower time instant bound              ::
%   ikeydir             Iscalar         Key for the zero crossing direction   ::
%   epsrel              Rscalar         Relative error bound                  ::
%   epsabs              Rscalar         Absolute error bound                  ::
%                                                                             ::
% Output data dictionary:                                                     ::
%   tau                 Rscalar         Time instant of the zero crossing     ::
%   dudtau              Rscalar         Slope at the found time instant       ::
%                                                                             ::
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

%...............................................................................
%                                                                              :
% Compute auxiliary variables                                                  :
%..............................................................................:
inh = (length(uh)-1)/2;

%...............................................................................
%                                                                              :
% Built-up the time derivative matrix                                          :
%..............................................................................:
D = zeros(2*inh,1);
D(2:2:end) = 1:inh;
D = diag(D,1)-diag(D,-1);

%...............................................................................
%                                                                              :
% Check the highest harmonic coefficients for possible zero components         :
%..............................................................................:

% Compute the reference magnitude of the amplitude vector
uhampref = max(sqrt(uh(2:2:end-1).^2+uh(3:2:end).^2));

% Remove the highest harmonic coefficient if it is necessary
uhtmp = uh;
for ii = inh:-1:1
  uhamp = sqrt(uh(2*ii)^2+uh(2*ii+1)^2);
  if (uhamp < epsrel*uhampref)
    uhtmp(end-1) = [];
    uhtmp(end) = [];
  else
    break
  end
end

% Check the results
if (length(uhtmp) == 1)
  tau = [];
  if (nargout > 1)
    dudtau = [];
  end
  return
end

%...............................................................................
%                                                                              :
% Compute the roots of the equivalent algebraic problem                        :
%..............................................................................:

% Vector of the coefficients of the equivalent complex polynomial
h = flipud([uhtmp(end-1:-2:2)+1i*uhtmp(end:-2:3); 2*uhtmp(1); ...
            uhtmp(2:2:end-1)-1i*uhtmp(3:2:end)]);

% Compute the roots
z = roots(h);

% Check the results
if  (any(z == 0))
  error('ERROR: Error in the function ''func_myzerocrossing'' !');
end

%...............................................................................
%                                                                              :
% Calculate the real time instants of zero crossings from the computed roots   :
%..............................................................................:

% Select the roots lying on the unit circle
z = z(abs(abs(z)-1) < epsrel);
if (isempty(z))
  tau = [];
  if (nargout > 1)
    dudtau = [];
  end
  tautmp = 0;
  return
end

% Take the natural logarithm of the complex argument to get the time instants
tautmp = -real(1i*log(z));

% Compute the derivative at the found time instants of a possible zero crossing
dudtau = zeros(size(tautmp));
for ii = 1:length(tautmp)
  h = [1, zeros(1,2*inh)];
  h(2:2:end-1) = cos((1:inh)*tautmp(ii));
  h(3:2:end) = sin((1:inh)*tautmp(ii));
  dudtau(ii) = h*D*uh;
end

% Select only the zero crossings with the correct derivative
switch (ikeydir)

  case (1)
    tautmp = tautmp(dudtau > epsabs);
  case (0)  
    tautmp = tautmp(abs(dudtau) > epsabs);
  case (-1)
    tautmp = tautmp(dudtau < epsabs);

end

% Check the selected zero crossings
if (isempty(tautmp))
  tau = [];
  if (nargout > 1)
    dudtau = [];
  end
  return
end

%...............................................................................
%                                                                              :
% Shift and sort the computed real time instants                               :
%..............................................................................:
tautmp(tautmp < 0) = tautmp(tautmp < 0)+2*pi;
tautmp = sort(tautmp);

%...............................................................................
%                                                                              :
% Search for the needed time instant of the zero crossing                      :
%..............................................................................:
tautmp = [tautmp+fix(tau0/2/pi)*2*pi; tautmp+fix(tau0/2/pi)*2*pi+2*pi];
tau = min(tautmp(tautmp-tau0 > epsabs));

%...............................................................................
%                                                                              :
% Compute the slope at the found time instant                                  :
%..............................................................................:
if (nargout > 1)
  h = [1, zeros(1,2*inh)];
  h(2:2:end-1) = cos((1:inh)*tau);
  h(3:2:end) = sin((1:inh)*tau);
  dudtau = h*D*uh;
end

%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%                                                                             ::
% End of the function 'func_myzerocrossing'                                   ::
%                                                                             ::
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

