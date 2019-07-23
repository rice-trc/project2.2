function Sol = mhbm_postprocess(Y,fun_residual,fun_postprocess)

% Initialize output
Sol = struct;

% Obtain additional information about solution
if contains(func2str(fun_residual),'pnlss')
    [~,~,~,~,~,Xc,Om] = feval(fun_residual,Y);
    for ii=1:length(fun_postprocess)
        Sol = feval(fun_postprocess{ii},Sol,Xc,Om);
    end
else
    [~,dR_dX,~,Fnl,S,Fe,X,Om,Ceff] = feval(fun_residual,Y);
    % Evaluate user supplied postprocess functions
    for ii=1:length(fun_postprocess)
        Sol = feval(fun_postprocess{ii},Sol,X,Om,dR_dX,Fnl,S,Fe,Ceff);
    end
end
