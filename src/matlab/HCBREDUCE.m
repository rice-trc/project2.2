function [Mr,Kr,TFM] = HCBREDUCE(M,K,bdofs,ncomp, varargin)
%HCBREDUCE Returns the Fixed Interface Hurty/Craig-Bampton Reduced
%Mass and Stiffness Matrices for a given symmetric system.
% USAGE:
%	[Mr,Kr,TFM] = HCBREDUCE(M,K,bdofs,ncomp);
% INPUTS:
%   M,K		: NdofxNdof mass & stiffness matrices 
%   bdofs	: Nbx1 set of boundary/retained DOF's
%   ncomp	: 1x1 Number of fixed boundary modes
% OUTPUTS:
%   Mr,Kr	: (Ndof-Nb)x(Ndof-Nb) Reduced mass & stiffness
%   		  matrices
%   TFM		: Ndofx(Ndof-Nb) Transformation matrix

    Ndof = size(M,1);
    bdofs = reshape(bdofs, 1, length(bdofs));
    idofs = setdiff(1:Ndof,bdofs);

    Kbb = 0.5*(K(bdofs,bdofs)+K(bdofs,bdofs)');
    Kbi = 0.5*(K(bdofs,idofs)+K(idofs,bdofs)');
    Kii = 0.5*(K(idofs,idofs)+K(idofs,idofs)');

    Mbb = 0.5*(M(bdofs,bdofs)+M(bdofs,bdofs)');
    Mbi = 0.5*(M(bdofs,idofs)+M(idofs,bdofs)');
    Mii = 0.5*(M(idofs,idofs)+M(idofs,idofs)');
    clear K M

    if nargin==4  % No expected nulls
        PhiC = -Kii\(Kbi'); % Constraint modes
    else
%         [Vii, Dii] = eigs(Kii, varargin{1}*3, 'SM');  [Dii,si] = sort(diag(Dii));
%         Vii = Vii(:, si);
%         L = null(Vii(:, 1:varargin{1})');

        L = null( null(full(Kii))' );
        PhiC = -L* ((L'*Kii*L)\(L'*Kbi'));
    end
    try
        [PhiN,Dfb] = eigs(Kii,Mii,ncomp*2,'SM');
    catch
        fprintf('Attemping full eigen solution\n');
        [PhiN,Dfb] = eig(full(Kii),full(Mii));
    end
    [Dfb,si] = sort(diag(Dfb));
    PhiN = PhiN(:,si);  PhiN = PhiN./sqrt(diag(PhiN'*Mii*PhiN)');
    Dfb = Dfb((1:ncomp));
    PhiN = PhiN(:,(1:ncomp));
    
    Mcc = Mbb+Mbi*PhiC+PhiC'*Mbi'+PhiC'*Mii*PhiC;
    Mcn = (Mbi+PhiC'*Mii)*PhiN;
  
    Mr = [Mcc Mcn; Mcn' eye(ncomp)];%PhiN'*Mii*PhiN];
    Kr = [Kbb+Kbi*PhiC zeros(size(Mcn)); 
        zeros(size(Mcn')) diag(Dfb)];%PhiN'*Kii*PhiN];
    TFM = sparse([eye(length(bdofs)) zeros(length(bdofs),ncomp);
           PhiC PhiN]);
    TFM([bdofs idofs], :) = TFM;
end