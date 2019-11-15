clc
clear all
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')
addpath('../src/matlab/')
load('./data/fresp.mat', 'Sols', 'exc_lev', 'oscillator', 'PHI_L2')
Alevels = [0.01 0.25 0.50 0.75]*1000;
% fs = 4096;
fs = 16384;

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 
set(0,'defaultAxesFontSize',13)

nx = [2 3];

%% Process NMA data
load('./data/nma.mat', 'X', 'H');
Psi = X(1:end-3,:);
om = X(end-2,:);
del = X(end-1,:);
l10a = X(end,:);
a = 10.^l10a;

p2 = (om.^2-2*(del.*om).^2)';
om4 = om'.^4;
Phi = Psi(1+(1:1),:)-1j*Psi(2+(1:1),:);
Fsc = (abs(Phi'*oscillator.Fex1)./a').^2;
mA = abs(oscillator.Fex1*a.*Phi)/sqrt(2);

%% From simulated experiments
load('./data/SimExp_shaker_yes_NMROM.mat', 'res_NMA'); %, 'a', 'p2', 'om4', 'Phi', 'Fsc', 'mA');
Psi = res_NMA.Phi_tilde_i;
om = res_NMA.om_i;
del = res_NMA.del_i_nl.*PHI_L2^2;
a = res_NMA.q_i/PHI_L2;

p2 = (om.^2-2*(del.*om).^2)';
om4 = om'.^4;
Phi = Psi;
Fsc = (abs(Phi'*oscillator.Fex1)./a').^2;
% Fsc = abs(res_NMA.F_i./a).^2
mA = abs(PHI_L2*a.*Phi)/sqrt(2);

%% Plotting Forced Responses
aa = gobjects(size(exc_lev));
bb = gobjects(3,1);
colos = distinguishable_colors(length(exc_lev));
for ia=[1:length(Alevels)]
    load(sprintf('./data/pnlssfresp_shaker_A%.2f_F%d_nx%s.mat',Alevels(ia), fs, sprintf('%d',nx)), 'Solspnlss');
    
    figure((ia-1)*2+1)
    clf()
%     figure((ia-1)*2+2)
%     clf()
    for iex=1:length(exc_lev)
        om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*exc_lev(iex)^2));   ris1 = find(imag(om1)==0);
        om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*exc_lev(iex)^2));   ris2 = find(imag(om2)==0);
                
        figure((ia-1)*2+1)
        aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-', 'Color', colos(iex,:)); hold on
        if iex==length(exc_lev)
            bb(1) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-', 'Color', colos(iex,:));
            bb(2) = plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2), '.-', 'Color', colos(iex,:))
            plot(om1(ris1)/2/pi, mA(ris1), '+:', 'Color', colos(iex,:));
            bb(3) = plot(om2(ris2)/2/pi, mA(ris2), '+:', 'Color', colos(iex,:));
        else
            plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2), '-', 'Color', colos(iex,:));
            plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2), '.-', 'Color', colos(iex,:))
            plot(om1(ris1)/2/pi, mA(ris1), '+:', 'Color', colos(iex,:));
            plot(om2(ris2)/2/pi, mA(ris2), '+:', 'Color', colos(iex,:));
        end
        
        legend(aa(iex), sprintf('F = %.2f N', exc_lev(iex)))

%         figure((ia-1)*2+2)
%         aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,3), '-', 'Color', colos(iex,:)); hold on
%         plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,3), '.-', 'Color', colos(iex,:));
    end
    figure((ia-1)*2+1)
    legend(aa(1:end), 'Location', 'southeast')
    xlim([200 350])
    set(gca, 'Position', [0.08 0.14 0.88 0.8])
    
    xlabel('Forcing frequency (Hz)')
    ylabel('RMS response amplitude (m)')
    
	aax=axes('position',get(gca,'position'),'visible','off');
    legend(aax, bb(1:3), 'HB', 'PNLSS','NM-ROM', 'Location','east')
    
    ax = axes('Position', [0.15 0.575, 0.3, 0.3])
	om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*exc_lev(3)^2));   ris1 = find(imag(om1)==0);
	om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*exc_lev(3)^2));   ris2 = find(imag(om2)==0);
    plot(Sols{3}(:,1)/2/pi, Sols{3}(:,2), '-', 'Color', colos(3,:)); hold on
    plot(Solspnlss{3}(:,1)/2/pi, Solspnlss{3}(:,2), '.-', 'Color', colos(3,:))
    plot(om1(ris1)/2/pi, mA(ris1), '+:', 'Color', colos(3,:));
	plot(om2(ris2)/2/pi, mA(ris2), '+:', 'Color', colos(3,:));
    xlim([270 280]); ylim([2.5e-4 3.1e-4])
    
    print(sprintf('./extabs_fig/b1_shaker_fresp_comp_A%.2f_nx%s.eps', Alevels(ia), sprintf('%d',nx)), '-depsc')
    
%     figure((ia-1)*2+2)
%     xlim([200 400])
%     set(gca, 'Position', [0.08 0.1 0.88 0.85])
%     xlabel('Forcing Frequency (Hz)')
%     ylabel('Response Phase (degs)')
end

%% Plotting Time Data (used to train PNLSS)

for ia=1:length(Alevels)
    load(sprintf('./data/ode45_multisine_shaker_A%.2f_F%d.mat',Alevels(ia), fs))
    
    figure(ia*10); clf()
    scatter_kde(y(:,1,1), ydot(:,1,1), '.', 'MarkerSize', 50)
    [n,c] = hist3([y(:,1,1),ydot(:,1,1)]);
    dxdy = prod([range(y(:,1,1)) range(ydot(:,1,1))]./size(n));    
    hold on;  contour(c{1}, c{2}, 0.5*(n/sum(n(:)))/dxdy, 'LineWidth', 2);
    yy=colorbar();
    ylabel(yy, 'pde')
    xlabel('y')
    ylabel('dy/dt')
    box on
    title(sprintf('A = %.2f N', Alevels(ia)))
    fprintf('done %d/%d\n',ia,length(Alevels))
    
    print(sprintf('./extabs_fig/b1_shaker_tdata_kern_A%.2f.eps',Alevels(ia)),'-depsc')
    close(ia*10)
end

%% Time data in frequency domain
figure(10)
clf()
aa = gobjects(size(Alevels));
bb = gobjects(2,1);
colos = distinguishable_colors(length(Alevels));
for ia=1:length(Alevels)
    load(sprintf('./data/ode45_multisine_shaker_A%.2f_F%d.mat',Alevels(ia), fs))
    
    y = y(:,end,1);
    u = sf(:,end,1);
    t = t(1:length(y));
    
    [freqs, yf] = FFTFUN(t', y);
    [freqs, uf] = FFTFUN(t', u);
    
    aa(ia) = semilogy(freqs, abs(yf), '.', 'Color', colos(ia,:)); hold on
    legend(aa(ia), sprintf('A = %.2f N', Alevels(ia)));
    
    if ia==4
        bb(1) = semilogy(freqs, abs(uf), '-', 'Color', colos(ia,:)); hold on
        bb(2) = semilogy(freqs, abs(yf), '.', 'Color', colos(ia,:)); hold on
        
        legend(bb(1:2), 'Excitation (N)', 'Response (m)')
    else
        semilogy(freqs, abs(uf), '-', 'Color', colos(ia,:)); hold on
    end
end
legend(aa(1:end), 'Location', 'southeast')
xlim([100 1100])
ylim([1e-10 1e2])
xlabel('Frequency (Hz)')
ylabel('Frequency content amplitude')

aax=axes('position',get(gca,'position'),'visible','off');
legend(aax, bb(1:2), 'Location', 'northeast');

print('./extabs_fig/b1_shaker_tdata_freqcont.eps', '-depsc')