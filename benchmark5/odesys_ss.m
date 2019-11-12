function dxdt = odesys_ss(t,x, u, model)
% system with NL.
% fex should either be a function or interpolated table

dxdt = model.A*x + model.B*u(t) + fnl(x(model.nx), x(model.nxd), model.nlpars);
end

function f = fnl(q,u,par)
% calculate nonlinear force
% q: displacement
% u: velocity

f = zeros(size(q,1),1);
nonlinear_elements = par.nonlinear_elements;
for nl=1:length(nonlinear_elements)
    % Determine force direction and calculate displacement and velocity of
    % nonlinear element
    w = nonlinear_elements{nl}.force_direction;
    qnl = w'*q; unl = w'*u;
    
%     switch lower(nonlinear_elements{nl}.type)
%     case 'unilateralspring'
    f = f + ...
        w*nonlinear_elements{nl}.stiffness*...
        (qnl-nonlinear_elements{nl}.gap).*...
        double(qnl-nonlinear_elements{nl}.gap>=0);
%     end
end
end
