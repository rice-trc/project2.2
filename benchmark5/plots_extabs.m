clc
clear all
addpath('../src/nlvib/SRC/')
addpath('../src/nlvib/SRC/MechanicalSystems/')
addpath('../src/matlab/')

load('./Data/Fresp.mat', 'Sols', 'Fas', 'beam', 'beam_hcb', 'Fex1_hcb', 'Thcb', 'Ws', 'We')
beam_hcb.Fex1 = Fex1_hcb;
Alevels = [15, 30, 60, 120];

set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 
set(0,'defaultAxesFontSize',13)

nx = [2 3];
na = 2;

%% Process Simulated Experiments Data
load('./data/SimExp_shaker_no_NMROM.mat', 'res_NMA');
Psi = res_NMA.Phi_tilde_i;
om = res_NMA.om_i;
del = res_NMA.del_i_nl;
a = abs(res_NMA.q_i);

p2 = (om.^2-2*(del.*om).^2)';
om4 = om'.^4;
Phi = Psi;
Fsc = (abs(Phi'*beam.Fex1(1:2:end))./a').^2;

mA = abs(a.*Phi(res_NMA.options.eval_DOF))/sqrt(2);

%% Plotting forced responses
aa = gobjects(size(Fas));
bb = gobjects(3,1);

upsamp = 4;
dataname = 'ms_full';
load(sprintf('data/b%d_A%d_up%d_%s',5,Alevels(1),upsamp,dataname), 'fs')

pll = 0;

factive = 1:length(Fas);

colos = distinguishable_colors(length(Fas));
for ia=1:length(Alevels)
    Alevel = Alevels(ia);
    if ~pll
        load(sprintf('./data/pnlssfresp_A%.2f_F%d_nx%s.mat',Alevel,fs,sprintf('%d',nx)), 'Solspnlss');
    else
        error('No PLL PNLSS data available');
    end
    
	figure((ia-1)*2+1)
    clf()
    factive = 1:length(Solspnlss);
	for iex=factive
        om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*Fas(iex)^2));   ris1 = find(imag(om1)==0);
        om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*Fas(iex)^2));   ris2 = find(imag(om2)==0);
        ris1 = find(abs(angle(om1))<0.001);
        ris2 = find(abs(angle(om2))<0.001);
        
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
    xlim([30 80])
%     xlim(sort([Ws We])/2/pi)
%     ylim([1e-5 1e0])
    
    xlabel('Forcing frequency (Hz)')
    ylabel('RMS response amplitude (m)')
    
    legend(aa(factive), 'Location', 'northeast')
    
	aax=axes('position',get(gca,'position'),'visible','off');
    legend(aax, bb(1:3), 'Location', 'northwest');
    
    if ~pll
        print(sprintf('./extabs_fig/b5_fresp_comp_A%.2f_nx%s.eps',Alevel,sprintf('%d',nx)), '-depsc')
    else
        error('No PLL PNLSS data available');
%         print(sprintf('./extabs_fig/b4_fresp_comp_pll_%s_nx%s.eps',fdirs{ia},sprintf('%d',nx)), '-depsc')
    end
end

%% Looking at Time Data (used to train PNLSS)
upsamp = 4;
dataname = 'ms_full';

for ia=1:length(Alevels)
    load(sprintf('data/b5_A%d_up%d_%s',Alevels(ia),upsamp,dataname))
    fdof = find(beam.Fex1);
    
    figure(ia*10); clf()
    h=scatter_kde(y(:,1,1,fdof), ydot(:,1,1,fdof), '.', 'MarkerSize', 50);
    [n,c] = hist3([y(:,1,1,fdof), ydot(:,1,1,fdof)]);
    dxdy = prod([range(y(:,1,1,fdof)) range(ydot(:,1,1,fdof))]./size(n));
    hold on; contour(c{1}, c{2}, 0.5*(n/sum(n(:)))/dxdy, 'LineWidth', 2);
    yy=colorbar();
    yy.Limits = [min(h.CData) max(h.CData)];
    ylabel(yy, 'pde')
    xlabel('y')
    ylabel('dy/dt')
    box on
    title(sprintf('A = %.2f N', Alevels(ia)))
    fprintf('done %d/%d\n', ia, length(Alevels))
    
    print(sprintf('./extabs_fig/b5_tdata_kern_A%.2f.eps',Alevels(ia)), '-depsc')
    close(ia*10)
end

%% Looking at frequency response in Time data (used to train PNLSS)
upsamp = 4;
dataname = 'ms_full';

figure(10)
clf()
aa = gobjects(size(Alevels)); 
bb = gobjects(2,1);
colos = distinguishable_colors(length(Alevels));
for ia=1:length(Alevels)
    load(sprintf('data/b5_A%d_up%d_%s',Alevels(ia),upsamp,dataname), 't', 'y', 'u', 'fs')
    fdof = find(beam.Fex1);
    
    y = reshape(y(:,end,1,fdof), [], 1);
    u = reshape(u(:,end,1), [], 1);
    t = t(1:length(y));
    
    [freqs, yf] = FFTFUN(t', y);
    [freqs, uf] = FFTFUN(t', u);
   
    aa(ia) = semilogy(freqs, abs(yf), '.', 'Color', colos(ia,:)); hold on
    legend(aa(ia), sprintf('A = %.2f N', Alevels(ia)))
    
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
ylim([1e-8 1e1])
xlabel('Frequency (Hz)')
ylabel('Frequency content amplitude')

aax=axes('position',get(gca,'position'),'visible','off');
legend(aax, bb(1:2), 'Location', 'northeast');

print('./extabs_fig/b5_tdata_freqcont.eps','-depsc')