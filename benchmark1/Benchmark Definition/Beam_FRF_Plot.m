clear; clc;
close all; 
set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

load FlatBeam_FRF.mat

%% Illustrate phase diagram
figure(1)
hold on
for iex=1:length(exc_lev)
    plot(Om{iex}/2/pi, phase{iex}, '-', 'LineWidth', 2);
end

for iex=1:length(exc_lev)
    res{iex} = interp1(phase{iex},Om{iex}/2/pi,-90);
    plot(res{iex},-90,'ok', 'LineWidth', 2)  % find resonance point
end

xlabel('Forcing Frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
legend('Fex = 10','Fex = 40','Fex = 60','Fex = 80','Fex = 100',...
    'Location','E');
hold off

%% Illustrate frequency response
figure (2); hold on; box on
for iex=1:length(exc_lev)
plot(Om{iex}/2/pi,a_w_L_2{iex},'linewidth',1.5);
end
plot(om_HB/2/pi,abs(a_w_L_2_NMA),'--k','linewidth',2)
legend('Fex = 10','Fex = 40','Fex = 60','Fex = 80','Fex = 100','EPMC',...
    'Location','E');
for iex=1:length(exc_lev)
peak_amp = interp1(Om{iex}/2/pi,a_w_L_2{iex},res{iex}); % mark resonance point
plot(res{iex},peak_amp,'ok', 'LineWidth', 2)
end
xlabel('Forcing Frequency $\omega$ (Hz)')
ylabel('RMS Response Displacement Amplitude (m)')
xlim([100 650])

plot(om_HB/2/pi,abs(a_w_L_2_NMA),'--k','linewidth',2)
legend('Fex = 10','Fex = 40','Fex = 60','Fex = 80','Fex = 100','EPMC',...
    'Location','E');
hold off

%% Illustrate Damping Ratio
figure (3)
Y_HB_1 = Q_HB(n+(1:n),:)-1i*Q_HB(2*n+(1:n),:);
semilogx(abs(Y_HB_1),del_HB,'b-','LineWidth',2);
% semilogx(1000*abs(a_w_L_2_NMA),del_HB*1e2,'b-','LineWidth',2);
xlabel('Modal Amplitude'); ylabel('Damping Factor')

%% Illustrate Backbone Curve
figure (4)
semilogx(1000*abs(a_w_L_2_NMA),om_HB/2/pi,'linewidth',2)
xlabel('Modal Amplitude'); ylabel('Natural Frequency (Hz)')

%% Illustrate Synthesized Frequency Response
figure
p2 = (om_HB.^2-2*(del_HB.*om_HB).^2)';
om4 = (om_HB.^4)';
Phi_HB = X_NM(n+(1:n),:)-1j*X_NM(2*n+(1:n),:);
Fsc = (abs(Phi_HB'*Fex1)./a_NMA').^2;
mAmps = abs(sqrt(0.5)*a_NMA.*(PHI_L_2*Phi_HB));

colos = distinguishable_colors(length(exc_lev));
livs = 10;
for k=1:length(exc_lev)
    aa(k+1) = semilogy(Om{k}/(2*pi), a_w_L_2{k}, ...
        '-', 'LineWidth', 2, 'Color', colos(k,:)); hold on
end
for k=1:length(exc_lev)
    om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*exc_lev(k)^2));
    om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*exc_lev(k)^2));

    ris1 = find(imag(om1)==0);  % real indices
    ris2 = find(imag(om2)==0);  % real indices
    semilogy(om1(ris1)/(2*pi), mAmps(ris1), '--', 'Color', colos(k,:)); hold on
    semilogy(om2(ris2)/(2*pi), mAmps(ris2), '--', 'Color', colos(k,:)); hold on
    
    ivs = fix(linspace(1, length(ris1), livs));
	semilogy(om1(ris1(ivs))/(2*pi), mAmps(ris1(ivs)), ...
        'o', 'Color', colos(k,:), 'MarkerFaceColor', colos(k,:)); hold on
    ivs = fix(linspace(1, length(ris2), livs));
    semilogy(om2(ris2(ivs))/(2*pi), mAmps(ris2(ivs)), 'o', ...
        'Color', colos(k,:), 'MarkerFaceColor', colos(k,:)); hold on
end

legend('Fex = 10','Fex = 40','Fex = 60','Fex = 80','Fex = 100',...
    'Location','E');
xlabel('Forcing Frequency $\omega$ (Hz)')
ylabel('RMS Response Displacement Amplitude (m)')
xlim([100 650])
hold off

%% Illustrate Synthesized Phase Diagram
figure; hold on;

zts = del_HB;
oms = om_HB;
p2  = (oms.^2-2*(zts.*oms).^2)';
om4 = (oms.^4)';
Phi_HB = X_NM(n+(1:n),:)-1j*X_NM(2*n+(1:n),:);
Fsc = (abs(Phi_HB'*Fex1)./a_NMA').^2;
phf = Phi_HB'*Fex1;

colos = distinguishable_colors(length(exc_lev));
livs = 10;
for k=1:length(exc_lev)
    aa(k+1) = semilogy(Om{k}/(2*pi), phase{k}, ...
        '-', 'LineWidth', 2, 'Color', colos(k,:)); hold on
end

for k=1:length(exc_lev)
    om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*exc_lev(k)^2));
    om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*exc_lev(k)^2));

    ris1 = find(imag(om1)==0);  % real indices
    ris2 = find(imag(om2)==0);  % real indices
    
    phi1 = rad2deg(angle(phf(ris1)./((oms(ris1)'.^2-om1(ris1).^2)+1j*(2*zts(ris1).*oms(ris1).*om1(ris1)')')));
    phi2 = rad2deg(angle(phf(ris2)./((oms(ris2)'.^2-om2(ris2).^2)+1j*(2*zts(ris2).*oms(ris2).*om2(ris2)')')));
    
    plot(om1(ris1)/2/pi, phi1, '--', 'Color', colos(k,:))
    plot(om2(ris2)/2/pi, phi2, '--', 'Color', colos(k,:))
    
    ivs = fix(linspace(1, length(ris1), livs));
    plot(om1(ris1(ivs))/2/pi, phi1(ivs), '.', 'Color', colos(k,:), 'MarkerSize', 20)
    ivs = fix(linspace(1, length(ris2), livs));
    plot(om2(ris2(ivs))/2/pi, phi2(ivs), '.', 'Color', colos(k,:), 'MarkerSize', 20)
end
xlabel('Forcing Frequency $\omega$ (Hz)')
ylabel('Response phase (degs)')
legend('Fex = 10','Fex = 40','Fex = 60','Fex = 80','Fex = 100',...
    'Location','E');
hold off