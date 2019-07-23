function [Xdot, z, fx, fy, fn] = ROC_DYNSYS(t, X, z, A, b, bx, by, bn, fricts, fex, frictfun)
%ROC_DYNSYS returns the rate of change of the states of the
%dynamical system with a frictional nonlinearity
% USAGE:
%   [Xdot] = ROC_DYNSYS(t, X, z, A, b, bx, by, bn, fricts, fex, frictfun);
% INPUTS:
%   t           : 1x1 time scalar
%   X           : Nsx1 state vector
%   A           : NsxNs linear state matrix
%   b           : Nsxfex.nf forcing coefficient matrix
%   bx,by,bn    : Nsxfricts.nel frictional force coefficient matrices
%   fricts      : Frictional parameters structure with the following,
%           nst	: Number of frictional states
%           nel	: Number of frictional elements
%       txdofs  : nelx1 X-tangential dofs
%       tydofs  : nelx1 Y-tangential dofs
%       ndofs   : nelx1 normal dofs
%       txwgts  : Weights for txdofs (1.0 or 0.0 to
%               	activate/deactivate)
%       tywgts  : Weights for tydofs (1.0 or 0.0 to
%                   activate/deactivate)
%       nwgts   : Weights for tydofs (1.0 or 0.0 to
%               	activate/deactivate)
%       sxdof   : nelxnst transformation from state vector to x
%           		tangential frict force
%       sydof   : nelxnst transformation from state vector to y
%               	tangential frict force
%       sndof   : nelxnst transformation from state vector to n
%                   normal force
%       ktx     : nelx1 x-tangential stiffnesses
%       kty     : nelx1 y-tangential stiffnesses
%       kn      : nelx1 normal stiffnesses
%       mu      : nelx1 Coefficients of friction
%       N0      : nelx1 Normal forces at dormant state
%   fex		: Structure with periodic (cosine) forcing fields,
%       nf	: Number of forcing points
%       dofs: nfx1 dofs being forced
%       fval: nfx1 forcing amplitudes
%       fpar: nfx1 forcing parameters
%       ffun: nfx1 cell with @(t) forcing function handles
%   frictfun: function handle of the form,
%		    [z, fx, fy, fn] = frictfun(t, X, z, fricts)
% OUTPUTS:
%   Xdot	: Nsx1 state velocity vector
%   z       : Nsx1 updated frictional state vector
%   fx,fy,fn: fricts.nelx1 frictional forces

    ft = cellfun(@(c) c(t), fex.ffun);
    Xdot = A*X + b*ft;
    
    [z, fx, fy, fn] = frictfun(t, X, z, fricts);
    Xdot = Xdot + bx*fx + by*fy + bn*fn;
end