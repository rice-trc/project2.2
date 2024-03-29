function dxdt = odesys(t,x, u, par)
% system with NL.
% fex should either be a function or interpolated table

M = par.M; D=par.D; K=par.K; amp=par.Fex1;

n = size(M,1);
y = x(1:n);
ydot = x(n+1:end);
dxdt = zeros(size(x));

dxdt(1:n) = ydot;
dxdt(n+1:end) = M\(amp*u(t) - K*y - D*ydot - fnl(y, ydot, par));
end

function fnl = fnl(q,u,par)
% calculate nonlinear force
% q: displacement
% u: velocity

fnl = zeros(size(q,1),1);
nonlinear_elements = par.nonlinear_elements;
for nl=1:length(nonlinear_elements)
    % Determine force direction and calculate displacement and velocity of
    % nonlinear element
    w = nonlinear_elements{nl}.force_direction;
    qnl = w'*q; unl = w'*u;
    
    switch lower(nonlinear_elements{nl}.type)
        case 'unilateralspring'
            fnl = fnl + ...
                w*nonlinear_elements{nl}.stiffness*...
                (qnl-nonlinear_elements{nl}.gap).*...
                double(qnl-nonlinear_elements{nl}.gap>=0);
        case 'tanhdryfriction'
            fnl = fnl + ...
                w*(nonlinear_elements{nl}.friction_limit_force.*...
                tanh(unl./nonlinear_elements{nl}.eps));
%         case 'elasticdryfriction': Hysteretic nonlinearities require
%         special treatment!
        otherwise
            error(['Unknown nonlinear element ' ...
                nonlinear_elements{nl}.type '.']);
    end
end
end