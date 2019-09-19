function Sol = mhbm_postprocess(Y,fun_residual,fun_postprocess,H,d,fs)

% Initialize output
Sol = struct;

% Obtain additional information about solution
if contains(func2str(fun_residual),'pnlss')
    [~,dR_dX,~,~,~,Xc,Om] = feval(fun_residual,Y);
    for ii=1:length(fun_postprocess)
        Sol = feval(fun_postprocess{ii},Sol,Xc,Om);
    end
    
    % Stability
    Dlmx = eye(2*H+1);   % Time delay matrix
    for n=1:H
        Dlmx(1+(2*n-1:2*n), 1+(2*n-1:2*n)) = [cos(n*Y(end)/fs) sin(n*Y(end)/fs);...
                                     -sin(n*Y(end)/fs) cos(n*Y(end)/fs)];
    end
    Dlmx = kron(Dlmx, eye(d));
    Ktil = dR_dX(:,1:end-1)+Dlmx;
    
%     Sol.mapmults = eig(dR_dX(:,1:end-1));
    Sol.mapmults = eig(Dlmx'*Ktil);
    lam = log(Sol.mapmults)*fs;
%     lam = -log(eig(dR_dX(1:d,1:d)))*fs;  % Using only 0 harmonics
    [~, si] = sort(imag(lam));
    Sol.hillsexps = lam(si(1:d));
    Sol.mapmults = Sol.mapmults(si(1:d));
%     Sol.hillsexps = lam(si(1:end));
%     Sol.mapmults = Sol.mapmults(1:end);
    
    Sol.hstab = isempty(find(abs(rad2deg(angle(Sol.hillsexps)))<89, 1));
%     Sol.hstab = isempty(find(abs(Sol.mapmults)>1, 1));
    if Sol.hstab
        Sol.stab = 1;
        Sol.unstab = nan;
    else
        Sol.stab = nan;
        Sol.unstab = 1;
    end

else
    [~,dR_dX,~,Fnl,S,Fe,X,Om,Ceff] = feval(fun_residual,Y);
    % Evaluate user supplied postprocess functions
    for ii=1:length(fun_postprocess)
        Sol = feval(fun_postprocess{ii},Sol,X,Om,dR_dX,Fnl,S,Fe,Ceff);
    end
end

end
