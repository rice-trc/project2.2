%========================================================================
% DESCRIPTION: 
% Mechanical system governed by the equations of motion
% 
%   M*\ddot q + D*\dot q + K*q + fnl(q) = fex
% 
%       where fnl(q) is a polynomial in q (stiffness nonlinearity),
% 
%       fnl(i) = \sum_k E(k,i) * z_k,   where E are the coefficients and 
%       z_k = \prod_j q_j^{p(k,j}       with nonnegative integers p are the
%                                       base polynomials.
%========================================================================
% This file is part of NLvib.
% 
% If you use NLvib, please refer to the paper:
%       M. Krack, J. Gross: The Harmonic Balance Method and its application 
%       to nonlinear vibrations: Introduction and current state of the art. 
%       submitted to Mechanical Systems and Signal Processing, 2018,
%       https://www.ila.uni-stuttgart.de/nlvib/downloads/HB_Krack.pdf.
% 
% COPYRIGHT AND LICENSING: 
% NLvib Version 1.0 Copyright (C) 2017  Malte Krack  
%										(krack@ila.uni-stuttgart.de) 
%                     					Johann Gross 
%										(gross@ila.uni-stuttgart.de)
%                     					University of Stuttgart
% This program comes with ABSOLUTELY NO WARRANTY. 
% NLvib is free software, you can redistribute and/or modify it under the
% GNU General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% For details on license and warranty, see http://www.gnu.org/licenses
% or gpl-3.0.txt.
%========================================================================
classdef System_with_PolynomialStiffnessNonlinearity < MechanicalSystem
    methods
        function obj = System_with_PolynomialStiffnessNonlinearity(...
                M,D,K,p,E,Fex1)
            %% Constructor
            
            % Handle incomplete input
            if nargin<=4
                Fex1 = zeros(length(M),1);
            end
            
            % Define nonlinearity
            nonlinearity = {struct('type','polynomialStiffness',...
                'exponents',p,'coefficients',E,'islocal',0)};
            
            % Call parent class constructor
            obj = obj@MechanicalSystem(M,D,K,...
                nonlinearity,Fex1);
        end
    end
end