function dxdt = odesys(t,x, par)
% system with NL.
% fex should either be a function or interpolated table

M = par.M; C=par.C; K=par.K; p=par.p; E=par.E; fex=par.fex; amp=par.amp;

n = size(M,1);
y = x(1:n);
ydot = x(n+1:end);
dxdt = zeros(size(x));

dxdt(1:n) = ydot;
dxdt(n+1:end) = M\(amp*fex(t) - K*y - C*ydot - fnl(y, ydot, p, E));
end

function f = fnl(y,ydot,p,E)
% polynomial stiffness NL; eq. C.2 p 143 in Malte K. HBnlvib book.

n = size(y,1);
nz = size(E,1);
f = E'*prod(kron(y',ones(nz,1)).^p,2);

% f = zeros(n,1);
% for i=1:n
%     Et = 0;
%     for k=1:nz
%         qt = 1;
%         for j=1:n
%             qt = qt* y(j)^p(k,j);
%         end
%         Et = Et + E(k,i)*qt;
%     end
%     f(i) = Et;
% end

end