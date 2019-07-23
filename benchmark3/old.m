
% time points in period
Ntd = 1e3;

%% load data
% load data from mat-file if called from this script
% https://stackoverflow.com/a/35359201
stack=dbstack;
if numel(stack) == 1 || exist('frfdata','var') == 0
    frfdata = load('data/frf.mat');
    nmadata = load('data/nma.mat');
end

frf = process_frf(frfdata,Ntd);
nma = process_nma(nmadata,Ntd);

%% set default line style
% https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html
set(groot,'defaultAxesColorOrder',[1 0 0;0 1 0;0 0 1;0 0.5 0;0.75 0 0.75],...
    'defaultAxesLineStyleOrder','--|:|-')
%% remove default linestyle
set(groot,'defaultAxesLineStyleOrder','remove')
set(groot,'defaultAxesColorOrder','remove')

% % Illustrate frequency response
% figure; hold on; box on
% aa = gobjects(frf.nex,1);
% for j = 1:frf.nex
%     aa(j) = plot(frf.Om{j}/2/pi,frf.apeak{j}*1000,'k-','linewidth',1.5);
%     legend(aa(j), sprintf('F = %.2f N', frf.exc_lev(j)));
% end
% legend(aa, 'Location', 'northeast')
% xlabel('$f_{\mathrm{ex}}$ in Hz');
% ylabel('$\hat{w}_{L/2}$ in mm');
% title(['thickness = ' num2str(thickness*1000) 'mm']);



% % Determine total energy in the system from the displacement and velocity
% % at t=0
% energy = zeros(size(a_NMA));
% % loop over countinuation parameter(omega)
% for i=1:length(nma.Om)
%     Qi = reshape(Q_HB(:,i),n,2*H+1);
%     q0 = Qi(:,1) + sum(Qi(:,2:2:end),2);
%     u0 = sum(Qi(:,3:2:end),2)*om_HB(i);
%     energy(i) = 1/2*u0'*beam.M*u0 + 1/2*q0'*beam.K*q0 + knl*q0(n-1)^4/4;
% end


%% FEP

% % Modal frequency vs. amplitude
% figure; hold on;
% plot(log10(energy),Om_nma/(2*pi),'k-o');
% xlabel('log10(energy)'); ylabel('modal frequency in Hz');
% % set(gca,'ylim',[20 50]);


%% NM-ROM

% % closed-from expression of frequency response for NM-ROM
% tau = linspace(0,2*pi,Ntd+1); tau = tau(1:end-1);
% 
% Nmod = nma.n;
% Fex1 = nma.gam;
% p2 = (nma.Om.^2-2*(nma.xi.*nma.Om).^2)';
% om4 = (nma.Om.^4)';
% Phi = nma.Psi(Nmod+(1:Nmod),:) - 1j*nma.Psi(2*Nmod+(1:Nmod),:);
% Fsc = (abs(Phi'*Fex1)./nma.a').^2;
% mAmps = nma.a.*range(real(exp(1j*tau(:))*(nma.PHI_L2*Phi)))/2;
% 
% livs = 5;
% for j = 1:frf.nex
%         % closed form solution
%     om1 = sqrt(p2 + sqrt(p2.^2-om4+Fsc*frf.exc_lev(j)^2));
%     om2 = sqrt(p2 - sqrt(p2.^2-om4+Fsc*frf.exc_lev(j)^2));
%     
%     % find real solution/indicies
%     rid1 = find(imag(om1)==0);
%     rid2 = find(imag(om2)==0);
%     plot(om1(rid1)/(2*pi), mAmps(rid1), '--', 'Color', colors(j,:));
%     plot(om2(rid2)/(2*pi), mAmps(rid2), '--', 'Color', colors(j,:));
%     
%     ivs = fix(linspace(1, length(rid1), livs));
%     plot(om1(rid1(ivs))/(2*pi), mAmps(rid1(ivs)),...
%         'o', 'Color', colors(j,:), 'MarkerFaceColor', colors(j,:));
%     ivs = fix(linspace(1, length(rid2), livs));
%     plot(om2(rid2(ivs))/(2*pi), mAmps(rid2(ivs)),...
%         'o', 'Color', colors(j,:), 'MarkerFaceColor', colors(j,:));
% end
    
%% NFRC

% % peak amplitude
% figure;
% hold on
% plot(nma.Om/2/pi,1000*abs(nma.apeak),'color',[1 .2 .3],'linewidth',2)
% 
% slabel = cell(1,frf.nex);
% for j = 1:frf.nex
%     plot(frf.Om{j}/2/pi,frf.apeak{j}*1000,'k-','linewidth',1.5);
%     slabel{j} = sprintf('F = %0.2f N',frf.exc_lev(j));
% end
% legend(['nm - backbone', slabel], 'Location', 'nw')
% 
% xlim([260 420])
% xlabel('$f_{\mathrm{ex}}$ in Hz');
% ylabel('$\hat{w}_{L/2}$ in mm');
% title(['thickness = ' num2str(thickness*1000) 'mm']);
% hfig(end+1) = {{gcf, "nfrc_peak"}};


%% bending modes

% % peak-peak value
% figure;
% plot(nma.Om/2/pi,nma.a_q,'linewidth',1.5)
% legend('$q_1$ (first bending)','$q_2$ (third bending)',...
%     '$q_3$ (fift bending)')
% xlabel('$\omega$ in Hz');
% ylabel('modal coordinates $q$ of NMA')
% xlim([min(nma.Om/2/pi) max(nma.Om/2/pi)]);
% title(['thickness = ' num2str(thickness*1000) 'mm']);
% hfig(end+1) = {{gcf, "bending_modes"}};


%%
Xm1 = X(1:oscillator.n:end-3, :);
Xm1c = Xm1([1 2:2:end], :) - 1j*[Xm1(1,:)*0; Xm1(3:2:end,:)];
Xm1a = abs(Xm1c);

figure(5);
clf()
aa = gobjects(H, 1);
for nh=1:H
    aa(nh) = semilogy(X(end-2,:)/2/pi, Xm1a(1+nh,:), '-'); hold on
    legend(aa(nh), sprintf('%d', nh))
end
legend(aa(1:nh))

%% HB convergence

cmap = distinguishable_colors(length(imod));

figure;
hold on
for i=imod
    hbc = load(sprintf('data/HConvDat_M%d.mat',i));
    
%     popt = {'Color',cmap(imod,:),'LineWidth', 2};
    h(i) = plot(hbc.Hs(1:end-2), hbc.Errs(1:end-2,1)*100, '-', 'Color',cmap(i,:),'LineWidth', 2);
    plot(hbc.Hs(1:end-2), hbc.Errs(1:end-2,2)*100, ':', 'Color',cmap(i,:),'LineWidth', 2)

    plot(hbc.Hs(hbc.ihconv), hbc.Errs(hbc.ihconv,:)*100, 'o',...
        'Color',cmap(i,:),'MarkerFaceColor', cmap(i,:))
    hline(hbc.errmax*100,'k--');  % treshold
    str{i} = sprintf('mode = %d', i);
end

% create lines for legend
L(1) = plot(nan, nan, 'k-','LineWidth', 2);
L(2) = plot(nan, nan, 'k:','LineWidth', 2);
L(3) = plot(nan, nan, 'ko','MarkerFaceColor','k');
L(4) = plot(nan, nan, 'k--');

set(gca, 'YScale', 'log')
xlabel('Number of Harmonics')
ylabel('Relative Error (\%)')
legend([h,L],{str{:},'Resonant frequency', 'Resonant amplitude',...
    'Convergence',sprintf('Threshold: %.2f \\%%', hbc.errmax*100) } )

%     ... 
%     sprintf('Convergence: $N_h$=%d', hbc.Nhconv),...
%     sprintf('Threshold: %.2f \\%%', hbc.errmax*100)} )

hfig(end+1) = {{gcf, "convergence"}};


