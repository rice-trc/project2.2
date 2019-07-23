function [A,B,C,T] = ss2phys(A,B,C)
% Calculate state space matrices in physical domain using a similarity
% transform T
%
% USAGE:
%  [A,B,C,T] = ss2phys(A,B,C)
% INPUTS:
%  State space matrices in continous form. Convert from discrete form by:
%  sys = d2c(ss(A,B,C,D,1/fs));
% OUTPUTS:
%  Transformed state space matrices and T
%
% See eq. (20.10) in
% Etienne Gourc, JP Noel, et.al
% "Obtaining Nonlinear Frequency Responses from Broadband Testing"
% https://orbi.uliege.be/bitstream/2268/190671/1/294_gou.pdf

% similarity transform
% mldivide (or \):  Solve systems of linear equations Ax = B for x
% Maybe this could be done without all the transposes with using mrdivide?

    T = [C; C*A];
%     C = mldivide(T.', C.').';        % (C = C*T^-1)
    C = C*inv(T);
%     A = mldivide(T.', (T * A).').';  % (A = T*A*T^-1)
    A = T*A*inv(T);
    B = T * B;

end