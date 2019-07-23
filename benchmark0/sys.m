function dxdt = sys(t,x, par)
% system with NL.
% fex should either be a function or interpolated table

M = par.M; C=par.C; K=par.K; fex=par.fex; amp=par.amp;

n = size(M,1);
y = x(1:n);
ydot = x(n+1:end);
dxdt = zeros(size(x));

dxdt(1:n) = ydot;
dxdt(n+1:end) = M\(amp*fex(t) - K*y - C*ydot);
end
