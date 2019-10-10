%========================================================================
% DESCRIPTION: 
% Class representing a mechanical system related to the equation of
% motion
% 
%           M*\ddot q + D*\dot q + K q + fnl = fex,
% 
%   where the nonlinear force assumes the form
% 
%               fnl = \sum_e w_e * fnl_e(w_e'*q,w_e'*\dot q).
% 
%  nonlinear_elements   structure array specifying nonlinear elements
%     .type             type (e.g. cubicSpring, unilateralSpring, ...)
%     .force_direction  n-vector 'w_e' with fnl = w_e*fnle(w_e'*q)
%     .[specif.prop.]   list of property-value combinations (e.g.
%                       stiffness, gap, ...)
% 
%       Fex1            excitation specification (optional)
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
classdef MechanicalSystem < handle
    properties
        n                   % number of degrees of freedom
        K                   % stiffness matrix
        M                   % mass matrix
        D                   % damping matrix
        nonlinear_elements  % nonlinear elements structure
        Fex1                % excitation structure
    end
    
    methods
        function obj = MechanicalSystem(M,D,K,...
                nonlinear_elements,Fex1)
            %% Constructor
            obj = obj@handle;
            
            % Store system matrices
            obj.M = M;
            obj.D = D;
            obj.K = K;
            obj.n = length(M);
            
            % Make sure nonlinear elements are in the expected cell form
            if ~isempty(nonlinear_elements) && ~iscell(nonlinear_elements)
                nonlinear_elements = {nonlinear_elements};
            end
            % Add nonlinearities
            obj.nonlinear_elements = nonlinear_elements;
            % Make sure the description of nonlinear elements is complete
            check_nonlinearities(obj);
            
            % Add excitation
            if nargin>4 && ~isempty(Fex1)
                if length(Fex1)~=obj.n
                    error(['Expecting fundamental harmonic excitation ' ...
                        'vector to have length ' num2str(obj.n) '.']);
                end
                obj.Fex1 = Fex1(:);
            else
                obj.Fex1 = zeros(obj.n,1);
            end
            
        end
        function check_nonlinearities(obj)
            % Add default information, if necessary
            for nl=1:length(obj.nonlinear_elements)
                if ~isfield(obj.nonlinear_elements{nl},'ishysteretic');
                    obj.nonlinear_elements{nl}.ishysteretic = 0;
                end
                if ~isfield(obj.nonlinear_elements{nl},'islocal');
                    obj.nonlinear_elements{nl}.islocal = 1;
                end
            end
        end
    end
end