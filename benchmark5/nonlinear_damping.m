function [ res_NMS ] = nonlinear_damping(res_LMA, res_NMA)
% Calculate nonlinear damping of system
%
% created M. Scheel, based on code of Simon Peter, INM
%
%  editing history
%  index	date		who		comment
%  ------------------------------------------------------------------------
% [001]     
%  ------------------------------------------------------------------------

Phi = res_LMA.Phi;

om_i = res_NMA.om_i; % measured undamped eigenfrequencies
Psi_tilde_i =  res_NMA.Psi_tilde_i; %nonlinear Eigenvectors (first order, not normalized, displacement)
P_act_1 = res_NMA.P_act_1;

% mass normalization of eigenvectors

M_exp = pinv(Phi')*pinv(Phi);
if det(M_exp)==0
    warning('Attention! Rank deficient experimental mass matrix!!!')
end
q_i = zeros(1,length(om_i));
Phi_tilde_i = zeros(size(Psi_tilde_i));
for ii = 1:length(om_i)
    q_i(ii) = sqrt(Psi_tilde_i(:,ii)'*M_exp*Psi_tilde_i(:,ii));
    Phi_tilde_i(:,ii) = Psi_tilde_i(:,ii)*(1/q_i(ii));
end

% -------------------------------------------------------------------------
%%%%%%%%%%% calculation of nonlinear damping
% -------------------------------------------------------------------------

m_i = q_i.^2;       
zeta = zeros(1,length(om_i));
for ii = 1:length(om_i)
    zeta(ii) = abs(P_act_1(ii))/(om_i(ii)^3*m_i(ii));
end

%Nichtlineare Dämpfung übergeben
res_NMS.del_i_nl = zeta;
res_NMS.Phi_tilde_i= Phi_tilde_i;
res_NMS.q_i=q_i;
res_NMS.M_exp=M_exp;
    
end
