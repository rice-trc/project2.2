%========================================================================
% DESCRIPTION: 
% Single mass oscillator with linear spring and damper, possible nonlinear
% elements and excitation.
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
classdef SingleMassOscillator < ChainOfOscillators
    methods
        function obj = SingleMassOscillator(m,d,k,...
                nonlinear_elements,varargin)
            % Maker sure force directions are given
            for i=1:length(nonlinear_elements)
                nonlinear_elements(i).force_direction = 1;
            end
            %% A single mass oscillator is just a special case of a 
            % chain of oscillators (add zero damping and stiffness to the
            % right)
            obj = obj@ChainOfOscillators(m,[d 0],[k 0],...
                nonlinear_elements,varargin{:});
        end
    end
end