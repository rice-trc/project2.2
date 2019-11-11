clc
clear all
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')
addpath('../src/matlab/')
load('./Data/Fresp.mat', 'Sols', 'Fas', 'beam')
beam.Fex1 = beam.Fex1/max(abs(beam.Fex1));
fdirs = {'famp001','famp01','famp05','famp08','famp20'}
% fdirs = {'famp001_n','famp01_n','famp05_n','famp08_n','famp20_n'}

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 
set(0,'defaultAxesFontSize',13)

nx = [2 3];
%% Numerical NMA data (to check)
load('./Data/NMA.mat', 'Xbb', 'Nh');
Psi = Xbb(1:end-3,:);
om = Xbb(end-2,:);
del = Xbb(end-1,:);
l10a = Xbb(end,:);
a = 10.^l10a;

p2 = (om.^2-2*(del.*om).^2)';
om4 = om'.^4;
Phi = Psi(beam.n+(1:beam.n),:)-1j*Psi(2*beam.n+(1:beam.n),:);
Fsc = (abs(Phi'*beam.Fex1)./a').^2;
mA = abs(a.*Phi(find(beam.Fex1),:))/sqrt(2);

%% Process PLL data
load('./Data/SimExp_shaker_no_NMROM.mat', 'res_NMA');
Psi = res_NMA.Phi_tilde_i;
om = res_NMA.om_i;
del = res_NMA.del_i_nl;
a = abs(res_NMA.q_i);

p2 = (om.^2-2*(del.*om).^2)';
om4 = om'.^4;
Phi = Psi;
Fsc = (abs(Phi'*beam.Fex1(1:2:end))./a').^2;
% Fsc = abs(res_NMA.F_i./a).^2

mA = abs(a.*Phi(res_NMA.options.eval_DOF))/sqrt(2);

%% Plotting forced response
aa = gobjects(size(Fas));
bb = gobjects(3,1);

pll = 0;

factive = 1:length(Fas);
% factive = [1 2 3 4 6];

colos = distinguishable_colors(length(Fas));
for ia=1:length(fdirs)
    if ~pll
        load(sprintf('./Data/pnlssfresp_%s_F500_nx%s.mat',fdirs{ia},sprintf('%d',nx)), 'Solspnlss');
    else
        load(sprintf('./Data/pnlss_pll_fresp_F%d_%s_nx%s.mat',500,fdirs{ia},sprintf('%d',nx)), 'Solspnlss');
    end
    
    figure((ia-1)*2+1)
    clf()
    factive = 1:length(Solspnlss);
    for iex=factive
        om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*Fas(iex)^2));   ris1 = find(imag(om1)==0);
        om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*Fas(iex)^2));   ris2 = find(imag(om2)==0);
        ris1 = find(abs(angle(om1))<0.01);
        ris2 = find(abs(angle(om2))<0.01);
        
        aa(iex) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2)/Fas(iex)*Fas(iex), '-', 'Color', colos(iex,:)); hold on
        if iex==1
            bb(1) = plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2)/Fas(iex)*Fas(iex), '-', 'Color', colos(iex,:)); hold on
            bb(2) = plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2)/Fas(iex)*Fas(iex), '--', 'Color', colos(iex,:));
            plot(om1(ris1)/2/pi, mA(ris1)/Fas(iex)*Fas(iex), '+:', 'Color', colos(iex,:));
            bb(3) = plot(om2(ris2)/2/pi, mA(ris2)/Fas(iex)*Fas(iex), '+:', 'Color', colos(iex,:));
            
            legend(bb(1), 'HB')
            legend(bb(2), 'PNLSS')
            legend(bb(3), 'NM-ROM')
        else
            plot(Sols{iex}(:,1)/2/pi, Sols{iex}(:,2)/Fas(iex)*Fas(iex), '-', 'Color', colos(iex,:)); hold on
            plot(Solspnlss{iex}(:,1)/2/pi, Solspnlss{iex}(:,2)/Fas(iex)*Fas(iex), '--', 'Color', colos(iex,:))
            plot(om1(ris1)/2/pi, mA(ris1)/Fas(iex)*Fas(iex), '+:', 'Color', colos(iex,:));
            plot(om2(ris2)/2/pi, mA(ris2)/Fas(iex)*Fas(iex), '+:', 'Color', colos(iex,:));
        end
        legend(aa(iex), sprintf('F = %.2f N', Fas(iex)))
    end
    set(gca, 'YScale', 'log')
    xlim([20 100])
    ylim([3e-8 3e-4])
    
    xlabel('Forcing frequency (Hz)')
    ylabel('RMS response amplitude (m)')
    
    legend(aa(factive), 'Location', 'northeast')
    
	aax=axes('position',get(gca,'position'),'visible','off');
    legend(aax, bb(1:3), 'Location', 'northwest');
    
    if ~pll
        print(sprintf('./extabs_fig/b4_fresp_comp_%s_nx%s.eps',fdirs{ia},sprintf('%d',nx)), '-depsc')
    else
        print(sprintf('./extabs_fig/b4_fresp_comp_pll_%s_nx%s.eps',fdirs{ia},sprintf('%d',nx)), '-depsc')
    end
end

%% Looking at Time Data (used to train PNLSS)
for ia=1:length(fdirs)
    load(sprintf('./TRANSIENT/%s/CLCLEF_MULTISINE.mat',fdirs{ia}), 'fsamp', 'famp', 'y', 'ydot', 'fdof');
    
    figure(ia*10); clf()
    scatter_kde(y(:,1,1,fdof), ydot(:,1,1,fdof), '.', 'MarkerSize', 50)
    [n,c] = hist3([y(:,1,1,fdof), ydot(:,1,1,fdof)]);
	dxdy = prod([range(y(:,1,1,fdof)) range(ydot(:,1,1,fdof))]./size(n));
    hold on; contour(c{1}, c{2}, 0.5*(n/sum(n(:)))/dxdy, 'LineWidth', 2);
    yy=colorbar();
    ylabel(yy, 'pde')
    xlabel('y')
    ylabel('dy/dt')
    box on
    title(sprintf('A = %.2f N', famp))
    fprintf('done %d/%d\n', ia, length(fdirs))
    
    print(sprintf('./extabs_fig/b4_tdata_kern_%s.eps',fdirs{ia}), '-depsc')
    close(ia*10)
end

%% Looking at frequency response in Time data (used to train PNLSS)
figure(10)
clf()
aa = gobjects(size(fdirs)); 
bb = gobjects(2,1);
for ia=1:length(fdirs)
    load(sprintf('./TRANSIENT/%s/CLCLEF_MULTISINE.mat',fdirs{ia}), 'fsamp', 'famp', 't', 'y', 'u', 'fdof');
    y = reshape(y(:,end,1,fdof), [], 1);
    u = reshape(u(:,end,1), [], 1);
    t = t(1:length(y));
    
    [freqs, yf] = FFTFUN(t', y);
    [freqs, uf] = FFTFUN(t', u);
   
    aa(ia) = semilogy(freqs, abs(yf), '.', 'Color', colos(ia,:)); hold on
    legend(aa(ia), sprintf('A = %.2f N', famp))
    
	if ia==4
        bb(1) = semilogy(freqs, abs(uf), '-', 'Color', colos(ia,:)); hold on
        bb(2) = semilogy(freqs, abs(yf), '.', 'Color', colos(ia,:)); hold on
        
        legend(bb(1:2), 'Excitation (N)', 'Response (m)')
    else
        semilogy(freqs, abs(uf), '-', 'Color', colos(ia,:)); hold on
    end
end
legend(aa(1:end), 'Location', 'east')
xlim([0 170])
ylim([1e-9 10^-.75])
xlabel('Frequency (Hz)')
ylabel('Frequency content amplitude')

aax=axes('position',get(gca,'position'),'visible','off');
legend(aax, bb(1:2), 'Location', 'northeast');

print('./extabs_fig/b4_tdata_freqcont.eps','-depsc')