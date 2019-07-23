function [Tmat] = NLCOEF_TFMMATS(Phi, Xpowers)
%NLCOEF_TFMMATS produces the transformation matrices necessary for
%transforming non-linear coefficient matrices
%  USAGE:
%   Tmat = NLCOEF_TFMMATS(Phi, Xpowers);
%  INPUTS:
%       Phi     :  nxn Transformation matrix (n states)
%                   Phi such that
%                   X = Phi*Q;
%       Xpowers :  nz x n Matrix of powers of each state per nonlinear term
%                   (nz nonlinear terms)
%  OUTPUTS:
%       Tmat    :  nzxnz matrix transforming nonlinear terms from physical
%                   to transformed domain
%                   g(X) = g(Phi*Q) = Tmat*g(Q)
    n  = size(Phi, 1);
    nz = size(Xpowers, 1);
    
%     % Without symbolics
%     pcofs = ones(n, max(max(Xpowers))+1);
%     Tmat = zeros(nz, nz);
%     for i=1:nz
%         for j=1:n
%             tmp = [1 zeros(1, Xpowers(i,j))];
%             pcofs(j, :)
%         end
%     end
    
    % With symbolics
    X = sym('x', [n, 1]);
    Q = sym('q', [n, 1]);
    Tmat = zeros(nz, nz);
    for i=1:nz
        [cq, tq] = coeffs(prod((Phi*Q).^(Xpowers(i,:)')), Q);  % coefficients and nonlinear terms in q
        
        for k=1:nz
            tqi = find(tq==prod(Q.^(Xpowers(k,:)')));
            if ~isempty(tqi)
                Tmat(i, k) = double(cq(tqi));
            end
        end
    end
end