addpath('../src/transient/')

% Adding spring mass and stiffness to FE beam.
% dof: transverse of 5'th node

Ne = 10;
Nn = Ne+1;

K = zeros(Nn*3);
M = zeros(Nn*3);
for e=1:Ne
    is = (e-1)*3+(1:6);
    
    [Me, Ke] = EBBEAM_MATS(7800, 200e9, 1.0, 1.0, 1.0);
    
    M(is, is) = M(is, is) + Me;
    K(is, is) = K(is, is) + Ke;
end
% dofs every node: (axial, transverse, gradient)

%% Permutation

L = eye(Nn*3);
L((5-1)*3+2,:) = [zeros(1,Nn*3-1) 1];
L(end,:) = [zeros(1,(5-1)*3+1) 1 zeros(1,Nn*3-((5-1)*3+2))];

LKL = L'*K*L;
LML = L'*M*L;

%% Adding spring mass
mt = 1.0;
kt = 1e9;

Mfull = blkdiag(LML,mt);
Kfull = blkdiag(LKL,0);  
Kfull(end-1:end,end-1:end) = Kfull(end-1:end,end-1:end) + kt*[1 -1;-1 1];