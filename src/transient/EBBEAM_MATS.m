function [Me, Ke] = EBBEAM_MATS(rho, E, A, I, L)
%EBBEAM_MATS returns the element matrices
%
% USAGE:
%   [Me, Ke] = EBBEAM_MATS(rho, E, A, I, L);
% INPUTS:
%   rho     : density
%   E       : Young's modulus
%   A       : Cross sectional area
%   I       : Second moment of cross section (about bending axis)
%   L       : Element length
% OUTPUTS:
%   Me      : 6x6 Element Mass matrix
%   Ke      : 6x6 Element Stiffness matrix

    Me = sparse(6,6);
    Ke = sparse(6,6);

    % Mass Matrix
    Me([1 4], [1 4]) = rho*A*L/6*[2 1; 1 2];  % Axial
    Me([2 3 5 6], [2 3 5 6]) = rho*A*L/420*...
        [156 22*L 54 -13*L;
        22*L 4*L^2 13*L -3*L^2;
        54 13*L 156 -22*L;
        -13*L -3*L^2 -22*L 4*L^2] + rho*I/(30*L)*...
        [36 3*L -36 3*L;
        3*L 4*L^2 -3*L -L^2;
        -36 -3*L 36 -3*L;
        3*L -L^2 -3*L 4*L^2];  % Bending
    
    % Stiffness Matrix
    Ke([1 4], [1 4]) = A*E/L*[1 -1;-1 1];  % Axial
    Ke([2 3 5 6], [2 3 5 6]) = 2*E*I/L^3*...
        [6 3*L -6 3*L;
        3*L 2*L^2 -3*L L^2;
        -6 -3*L 6 -3*L;
        3*L L^2 -3*L 2*L^2];  % Bending
end