function dxdt = sys_ss(t,x, model)
% state-space system with NL.
% fex should either be a function or interpolated table

A = model.A; B = model.B; C = model.C; D = model.D; fex=model.fex;
p = model.nlcof.power; E = model.nlcof.coef; fnls=model.nlcof.fnls;
nx=model.nlcof.nx; nxd=model.nlcof.nxd;
dxdt = zeros(size(x));

dxdt = A*x + B*fex(t) + fnls*fnl(x(nx),x(nxd),p,E);
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