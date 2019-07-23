function Sol = HB_hillsexp_nlvib(Y, fun_residual, M, C, Nh, Nd)
    
    zerM = sparse(size(M,1),size(M,2));
    Mtil = HARMONICSTIFFNESS(zerM,zerM,M,Y(end),0:Nh);
    Ctil = HARMONICSTIFFNESS(zerM,zerM,C,Y(end),0:Nh);
    [~,Ktil] = feval(fun_residual,Y);
    
    % Solving quadratic eigenvalue problem for hills' exponents
    Sol.hillsexps = polyeig(Ktil(:,1:end-1), Ctil, Mtil);
%     Sol.hillsexps = polyeig(Ktil(1:Nd,1:Nd), Ctil(1:Nd,1:Nd), Mtil(1:Nd,1:Nd));
    [~, si] = sort(abs(Sol.hillsexps));
    Sol.hillsexps = Sol.hillsexps(si(1:2*Nd));
    

    Sol.hstab = isempty(find(abs(rad2deg(angle(Sol.hillsexps)))<89, 1));
    if Sol.hstab
        Sol.stab = 1;
        Sol.unstab = nan;
    else
        Sol.stab = nan;
        Sol.unstab = 1;
    end
        

%     % Local positive definiteness of Jacobian
%     plot(real(Sol.hillsexps), imag(Sol.hillsexps), 'o')
%     
%     Sol.jaceigs = eig(Ktil);
%     Sol.jstab = isempty(find(Sol.jaceigs<0, 1));
end