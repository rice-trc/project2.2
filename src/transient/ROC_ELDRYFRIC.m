function [z, fx, fy, fn] = ROC_ELDRYFRIC(t, X, z, fricts)
%ROC_ELDRYFRIC returns the rate of change of tangential forces and
%the value of the normal force (since it's not state dependent) for
%given previous displacements and velocities
%
% USAGE:
%   [zdot, fn] = ROC_ELDRYFRIC(t, X, z, fricts);
% INPUTS:
%   t           : 1x1 time scalar
%   X           : (2*Nphx1) state vector    
%   z           : (fricts.nstx1) frictional states
%   fricts      : Frictional parameters structure with the following,
%           nst	: Number of frictional states
%           nel	: Number of frictional elements
%       txdofs  : nelx1 X-tangential dofs
%       tydofs  : nelx1 Y-tangential dofs
%       ndofs   : nelx1 normal dofs
%       txwgts  : Weights for txdofs (1.0 or 0.0 to
%               	activate/deactivate)
%       tywgts  : Weights for tydofs (1.0 or 0.0 to
%         			activate/deactivate)
%       nwgts   : Weights for tydofs (1.0 or 0.0 to
%               	activate/deactivate)
%       sxdof   : nelxnst transformation from state vector to x
%           		tangential frict force
%       sydof   : nelxnst transformation from state vector to y
%           		tangential frict force
%       sndof   : nelxnst transformation from state vector to n
%               	normal force
%       ktx     : nelx1 x-tangential stiffnesses
%       kty     : nelx1 y-tangential stiffnesses
%       kn      : nelx1 normal stiffnesses
%       mu      : nelx1 Coefficients of friction
%       N0      : nelx1 Normal forces at dormant state
% OUTPUTS:
%   z   	: fricts.nstx1 updated frictional states
%   fx		: fricts.nelx1 x-tangential force
%   fy		: fricts.nelx1 y-tangential force
%   fn		: fricts.nelx1 normal force

    Nph = fix(size(X,1)/2);
    ux 	  = fricts.txwgts*X(fricts.txdofs);
    uxdot = fricts.txwgts*X(Nph+fricts.txdofs);
    uy 	  = fricts.tywgts*X(fricts.tydofs);
    uydot = fricts.tywgts*X(Nph+fricts.tydofs);
    un 	  = fricts.nwgts*X(fricts.ndofs);
    undot = fricts.nwgts*X(Nph+fricts.ndofs);
    
    fn    = max(fricts.N0+fricts.kn.*un, 0);  % Normal force with
                                                             % compliant penalty
                                                             % penalty condition
    fn    = fn + (fn>0).*(fricts.cn*undot);  % Adding a normal dashpot when in contact
    
    % Stuck prediction
    fx    = (fricts.txwgts*fricts.ktx.*(ux-fricts.sxdof*z)).*(fn>0);
    fy    = (fricts.tywgts*fricts.kty.*(uy-fricts.sydof*z)).*(fn>0);
    Tf    = sqrt(fx.^2+fy.^2);   % Magnitude of Tangential friction
    fslip = max(fricts.mu.*fn, 0);
    fxslip = fslip.*fx./Tf;  fxslip(Tf==0) = 0.0;
    fyslip = fslip.*fy./Tf;  fyslip(Tf==0) = 0.0;
    
    % Slider Update
    z     = (fricts.sxdof'*((fricts.sxdof*z).*(fn>=0 & Tf<=fslip) + (ux-fxslip./fricts.ktx).*(fn>=0 & Tf>fslip) + (ux).*(fn<0))) + ...
        (fricts.sydof'*((fricts.sydof*z).*(fn>=0 & Tf<=fslip) + (uy-fyslip./fricts.kty).*(fn>=0 & Tf>fslip) + (uy).*(fn<0)));

    fx = (fricts.txwgts*fricts.ktx.*(ux-fricts.sxdof*z)).*(fn>0);
    fy = (fricts.tywgts*fricts.kty.*(uy-fricts.sydof*z)).*(fn>0);
    fn(fn>0) = fn(fn>0)-fricts.N0;
end