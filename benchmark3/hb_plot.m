% close all
% clearvars

srcpath = '../src/nlvib/SRC';
addpath(genpath(srcpath));
srcpath = '../src/matlab';
addpath(genpath(srcpath));

savefig = false;
% modes to plot
imod = [1,2,3];
benchmark = 3;

switch benchmark
    case 1
    case 2
    case 3
        % frf-plot limits for all considered modes
        ylm_frf = [0,1.25e-3; 0, 1.2e-4; 0, 2e-5];
        xlm_frf = [260 420; 1415, 1475; 3540,3595];
end

hfig = {};
%% use Latex
set(0,'defaultAxesTickLabelInterpreter', 'default');
set(0,'defaultTextInterpreter','latex'); 
set(0, 'DefaultLegendInterpreter', 'latex'); 

%% FRF, RMS amplitude

Ntd = 1024;
for i=imod

frfdata = load(sprintf('data/frf_M%d.mat',i));
nmadata = load(sprintf('data/nma_M%d.mat',i));

frf = process_frf(frfdata,Ntd);
nma = process_nma(nmadata,Ntd);
cmap = distinguishable_colors(frf.nex,'k');

figure;
hold on
ph = gobjects(frf.nex+1,1);
ph(1) = plot(nma.Om/2/pi,abs(nma.arms),'k-','linewidth',1.5,...
    'DisplayName','nm - backbone');

for j = 1:frf.nex
    ph(j+1) = plot(frf.Om{j}/2/pi,abs(frf.arms{j}),'-','Color',cmap(j,:),...
        'linewidth',1.5, ...
        'DisplayName',  sprintf('F = %0.2f N',frf.exc_lev(j)));
    % interpolated phase resonance
    plot(frf.pks(j,1)/2/pi, frf.pks(j,2),'o','Color',cmap(j,:), ...
        'MarkerFaceColor', cmap(j,:))
end
legend(ph,'Location', 'nw')

xlim(xlm_frf(i,:))
ylim(ylm_frf(i,:))
xlabel('Forcing Frequency $\omega$ (Hz)');
ylabel('$RMS \hat{w}_{L/2}$ (m)');
title(sprintf('mode %d',i));
hfig(end+1) = {{gcf, sprintf("nfrc_rms_M%d",i)}};

%% phase plot
figure;
hold on

ph = gobjects(frf.nex,1);
for j = 1:frf.nex
    ph(j) = plot(frf.Om{j}/2/pi,frf.phase{j}(end,:),'-',...
        'Color',cmap(j,:),'linewidth',1.5, ...
        'DisplayName',  sprintf('F = %0.2f N',frf.exc_lev(j)));
    % interpolated phase resonance
    plot(frf.pks(j,1)/2/pi,frf.pks(j,3),'o','Color',cmap(j,:), ...
        'MarkerFaceColor', cmap(j,:))
end
legend(ph, 'Location', 'ne')

xlim(xlm_frf(i,:))
xlabel('Forcing Frequency $\omega$ (Hz)');
ylabel('Response phase (degs)')
title(sprintf('mode %d',i));
hfig(end+1) = {{gcf, sprintf("nfrc_phase_M%d",i)}};

%% bending modes

figure;
plot(nma.Om/2/pi,nma.Qrms,'linewidth',1.5)
legend('$q_1$ (first bending)','$q_2$ (third bending)',...
    '$q_3$ (fift bending)', 'Location','nw')
xlabel('$\omega$ in Hz');
ylabel('modal coordinates $q$ of NMA')
xlim([min(nma.Om/2/pi) max(nma.Om/2/pi)]);
title(sprintf('mode %d',i));
hfig(end+1) = {{gcf,  sprintf("bending_modes_M%d",i)}};

%% NM freq-amp / freq-damp

figure;
plot(abs(nma.arms),nma.Om/2/pi,...
    'linewidth',1.5,'DisplayName',  sprintf('mode %d',i));
set(gca, 'XScale', 'log')

xlabel('Modal Amplitude');
ylabel('Natural Frequency (Hz)');
title(sprintf('mode %d',i));
hfig(end+1) = {{gcf, sprintf("nm_freq_M%d",i)}};

figure;
plot(abs(nma.arms),nma.xi*1e2,...
    'linewidth',1.5,'DisplayName',  sprintf('mode %d',i));
xlabel('Modal Amplitude');
ylabel('Damping Factor');
% set(gca, 'XScale', 'log', 'YScale', 'log')
set(gca, 'XScale', 'log')
title(sprintf('mode %d',i));
hfig(end+1) = {{gcf, sprintf("nm_damp_M%d",i)}};

end

%% COMBINED NM freq-amp/freq-damp
cmap = distinguishable_colors(length(imod),'k');

figure;
hfig(end+1) = {{gcf, "nm_freq"}};
h1=axes;
hold on
xlabel('Modal Amplitude');
ylabel('Natural Frequency (Hz)');

figure;
hfig(end+1) = {{gcf, "nm_amp"}};
h2=axes;
hold on
xlabel('Modal Amplitude');
ylabel('Damping Factor');

ph1 = gobjects(size(imod));
ph2 = gobjects(size(imod));
j=0;
for i=imod
j=j+1;
nmadata = load(sprintf('data/nma_M%d.mat',i));
nma = process_nma(nmadata,Ntd);

ph1(j) = plot(h1, abs(nma.arms),nma.Om/2/pi,'Color', cmap(j,:),...
    'linewidth',1.5,'DisplayName',  sprintf('mode %d',i));
ph2(j) = plot(h2, abs(nma.arms),nma.xi*1e2,'Color', cmap(j,:),...
    'linewidth',1.5,'DisplayName',  sprintf('mode %d',i));
end

legend(ph1)
legend(ph2)
set(h1, 'XScale', 'log')
set(h2, 'XScale', 'log', 'YScale', 'log')


%% HB convergence

for i=imod
    figure;
    hold on
    hbc = load(sprintf('data/HConvDat_M%d.mat',i));
    
    % plot both phase and amplitude convergence
    plot(hbc.Hs(1:end-2), hbc.Errs(1:end-2,:)*100, '-', 'LineWidth', 2);
    plot(hbc.Hs(hbc.ihconv), hbc.Errs(hbc.ihconv,:)*100, 'ko',...
        'MarkerFaceColor', 'k')
    hline(hbc.errmax*100,'k--');  % treshold

    title(sprintf('mode = %d', i))
    set(gca, 'YScale', 'log')
    xlabel('Number of Harmonics')
    ylabel('Relative Error (\%)')
    legend('Resonant frequency', 'Resonant amplitude', ...
        sprintf('Convergence: $N_h$=%d',hbc.Nhconv),...
        sprintf('Threshold: %.2f \\%%', hbc.errmax*100))

    hfig(end+1) = {{gcf, sprintf("convergence_M%d",i)}};
end


%% save figures

if savefig
    path = 'fig/';
    for i=1:length(hfig)
        h = hfig{i}{1};
        fname = hfig{i}{2};
        % change figure background from standard grey to white
        set(h, 'Color', 'w');
        export_fig(h, strcat(path,fname), '-pdf', '-png');
    end
    
end
