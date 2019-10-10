addpath('../src/transient/')

% Use regularized

% Adding spring mass and stiffness to FE beam.
% dof: transverse of 5'th node

% define system
% Properties of the beam
len = 2;                % length
height = .05*len;       % height in the bending direction
thickness = 3*height;   % thickness in the third dimension
E = 185e9;              % Young's modulus
rho = 7830;             % density
BCs = 'clamped-free';   % constraints

A = thickness*height;
I = thickness*height^3/12;

% number of equidistant nodes along length
Nn = 9;
Ne = Nn - 1;
% Nn = Ne+1;


K = zeros(Nn*3);
M = zeros(Nn*3);
for e=1:Ne
    is = (e-1)*3+(1:6);
    [Me, Ke] = EBBEAM_MATS(rho, E, A, I, len);
    M(is, is) = M(is, is) + Me;
    K(is, is) = K(is, is) + Ke;
end
% dofs every node: (axial, transverse, gradient)

% Attach additional mass and spring and apply dry friction element
% Slider attached to transverse dof (eg + 2)
slider_node = Nn;
slider_dof = (slider_node-1)*3 + 2;
kstatic = 3*E*I/len^3;
kt = kstatic*2e-1;
mt = 0.02 * M(slider_dof,slider_dof);

% Permutation.
L = eye(Nn*3);
L(slider_dof,:) = [zeros(1,Nn*3-1) 1];
L(end,:) = [zeros(1,(slider_node-1)*3+1) 1 zeros(1,Nn*3-((slider_node-1)*3+2))];

LKL = L'*K*L;
LML = L'*M*L;

% Adding spring mass
Mfull = blkdiag(LML,mt);
Kfull = blkdiag(LKL,0);
Kfull(end-1:end,end-1:end) = Kfull(end-1:end,end-1:end) + kt*[1 -1;-1 1];
