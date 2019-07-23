function pnlss_plot(t,sig,bla,estdata,valdata,testdata,valerrs,model,hfig,savefig,figname)

if nargin < 10
    savefig = false;
end
if nargin < 11
    figname = '';
end

% handle to figs. For saving plot
if ~exist('hfig','var')
    hfig = {};
end

freq = bla.freq; covY = bla.covY; fs = bla.fs;
P=sig.P; R=sig.R; Nt=sig.Nt; p=sig.p; m=sig.m;

%% modal analysis
% convert to continious time for modal analysis of linear part
sys_ct = d2c(ss(model.A,model.B,model.C,model.D,1/fs));
sd = modal(sys_ct.A,sys_ct.C);
wn = sprintf('%0.5g, ', sd.wn);
zeta = sprintf('%0.5g, ', sd.zeta);
disp('Identified Modal Parameters')
fprintf('Nat freq %s Hz. \ndamping %s\n',wn,zeta)

% similarity transform to get state space matrices in physical coordinates
% [Ap,Bp,Cp,T] = ss2phys(sys_ct.A,sys_ct.B,sys_ct.C);
% sys_phys = ss(Ap,Bp,Cp,model.D);


%% optimization path / convergence
[min_valerr,i] = min(valerrs);
figure;
plot(db(valerrs));
hold on
plot(i,db(min_valerr),'r.','Markersize',10)
xlabel('Successful iteration number')
ylabel('Validation error [dB]')
title('Selection of the best model on a separate data set')
hfig(end+1) = {{gcf, 'convergence'}};

%% Estimation data
cmap = distinguishable_colors(3,'k');

plottime = estdata;
plottime = reshape(plottime, [Nt*R,p,3]);
plotfreq = fft(reshape(plottime,[Nt,R,p,3]));
plotfreq = reshape(mean(plotfreq,2), [Nt,p,3]);
esterr_nl = plottime(:,:,3); % estdata(:,end-p+1:end);  % equal to yest-y_mod, []

% plot time series individual
for i=1:p
figure;
plot(t(1:Nt*R),squeeze(plottime(:,i,:)))
xlabel('Time (s)')
ylabel('Output (errors)')
legend('output','linear error','PNLSS error')
errstr = sprintf(['rms(y-y\\_mod) = %0.3e\n(Output error of the best',...
    ' PNLSS model on estimation data)\n'],rms(esterr_nl(:,i)) );
title(sprintf('Estimation results p: %d\n%s',p,errstr))
disp(' ')
disp(errstr)
% disp(['rms(noise(:))/sqrt(P) = ' num2str(rms(noise(:))/sqrt(P)) ' (= Noise level)'])
hfig(end+1) = {{gcf, sprintf('est_time_p%d',p)}};
end

% plot time series together
figure;
plot(t(1:Nt*R),reshape(plottime, [Nt*R, p*3]))
xlabel('Time (s)')
ylabel('Output (errors)')
legend('output','linear error','PNLSS error')
errstr = sprintf(['sum(rms(y-y\\_mod)) = %0.3e\n(Output error of the',...
    ' best PNLSS model on estimation data)\n'],sum(rms(esterr_nl,1)) );
title(['Estimation results' newline errstr])
disp(' ')
disp(errstr)
hfig(end+1) = {{gcf, sprintf('est_time')}};

% error in frequency domain
for i=1:p
figure;
hold on
plot(freq(1:end/2),db(squeeze(plotfreq(1:end/2,i,:))),'.') %,'Color',cmap)
plot(freq(1:end/2),db(sqrt(P*squeeze(covY(i,i,:)))), 'k.')
xlabel('Frequency (Hz)')
ylabel('Output (errors) (dB)')
legend('Output','Linear error','PNLSS error','noise')
title(sprintf('Estimation results p: %d',i))
hfig(end+1) = {{gcf, sprintf('est_err_p%d',i)}};
end

%% Validation data
plottime = valdata;
plottime = reshape(plottime, [Nt,p,3]);
plotfreq = fft(plottime);

for i=1:p
figure;
plot(freq(1:end/2),db(squeeze(plotfreq(1:end/2,i,:))),'.') % ,'Color',cmap(i,:))
xlabel('Frequency (Hz)')
ylabel('Output (errors) (dB)')
legend('Output','Linear error','PNLSS error')
title(sprintf('Validation results p: %d',i))
hfig(end+1) = {{gcf, sprintf('val_err_p%d',i)}};
end

%% Test, ie. newer seen data
plottime = testdata;
plottime = reshape(plottime, [Nt,p,3]);
plotfreq = fft(plottime);

for i=1:p
figure;
plot(freq(1:end/2),db(squeeze(plotfreq(1:end/2,i,:))),'.')
xlabel('Frequency');
ylabel('Output (errors) (dB)');
legend('Output','Linear error','PNLSS error')
title(sprintf('Test results p: %d',i))
hfig(end+1) = {{gcf, sprintf('test_err_p%d',i)}};
end

%% BLA plot. We can estimate nonlinear distortion
% total and noise distortion averaged over P periods and M realizations
% total distortion level includes nonlinear and noise distortion

for i=1:p
figure;
hold on;
plot(freq(bla.lines),db(abs(squeeze(bla.G(i,1,:)))))
plot(freq(bla.lines),db(abs(squeeze(bla.covGn(i,i,:)))*R,'power'),'s')
plot(freq(bla.lines),db(abs(squeeze(bla.covGML(i,i,:)))*R,'power'),'*')
xlabel('frequency (Hz)')
ylabel('magnitude (dB)')
title(sprintf('Estimated BLA p: %d)',i))
legend('BLA FRF','Noise Distortion','Total Distortion','Location','nw')
hfig(end+1) = {{gcf, sprintf('bla_p%d',i)}};
end

%% Compare BLA with subspace model
% GModel = fss2frf(A,B,C,D,freq(lines)/fs);
% G = bla.G;
% figure; hold on
% plot(freq(lines),db(abs(GModel(:))),'b-','LineWidth',4)
% plot(freq(lines),db(abs(G(:))),'r.')
% plot(freq(lines),db(abs(G(:)-GModel(:))),'c.')
% plot(freq(lines),db(abs(bla.covGML(:)),'power'),'k.')
% legend('BLA (parametric)','BLA (nonpar)','residual','standard deviation')

%% save figures

if savefig
    if ~isempty(figname)
        figname = strcat(figname,'_');
    end
    for i=1:length(hfig)
        h = hfig{i}{1};
        fname = hfig{i}{2};
        % change figure background from standard grey to white
        set(h, 'Color', 'w');
        export_fig(h, sprintf('./fig/%s%s',figname,fname), '-png');
    end
    
end


end