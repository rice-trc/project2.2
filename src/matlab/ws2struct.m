function S = ws2struct(varargin)
%% Collect variables in structure with corresponding fieldnames

vars = varargin;
for iv = 1:numel(vars)
    S.(vars{iv}) = evalin('caller', vars{iv});
end
end