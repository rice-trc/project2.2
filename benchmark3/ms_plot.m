function ms_plot(t,y,u,freq,lines,phi,benchmark,A,dataname,savefig)
% plot result from multisine excitation
% 1. time series
% 2. periodicity
% 3. force spectrum
% 4. detection lines. For random_odd excitation this shows if the
%    nonlinearity is odd/even or both
%
% phi is a vector with the mode factors, ie. converstion between modal and
% physical coordinate

if nargin < 10
    savefig = false;
end

% select realization to do the plotting for
r = 1;
% % select measurement point, ie. the mode factor in PHIS(np,:)
% np = 2;

[Nt,P,R,p] = size(y);


%% init plotting
hfig = {};

%% show time series
yres = phi*reshape(y(:,:,r,:),[],p)';

figure;
plot(t, yres,'k-')
% indicate periods
h1 = vline(t((1:r*P)*Nt),'--g');
% indicate realizations
h2 = vline(t((1:r)*Nt*P),'--k');set([h1 h2],'LineWidth',0.5)
xlabel('time (s)')
ylabel('magnitude')
title(['Multisine: ' num2str(r) ' realizations of ' num2str(P),...
    ' periods of ' num2str(Nt) ' samples per period'])

hfig(end+1) = {{gcf, "time"}};


%% show periodicity

figure;
per = (y(:,1:end-1,r)-y(:,end,r)) / rms(y(:,1,r));
plot(t(1:Nt*(P-1)),db(per(:)),'k-')
% indicate periods
h1 = vline(t((1:r*(P-1))*Nt),'--g');
% indicate realizations
h2 = vline(t((1:r)*Nt*(P-1)),'--k');set([h1 h2],'LineWidth',0.5)
xlabel('time (s)')
ylabel('Relative error to last period (dB)')
title([num2str(Nt) ' samples per period'])

hfig(end+1) = {{gcf, "periodicity"}};


%% show force signal for one period
% plot only half spectrum(single sided spectrum)
dft = fft(u(:,1,r));
nt = length(dft)/2+1;
xdft = dft(1:nt);

figure; subplot(2,1,1)
plot(freq(1:nt),db(xdft),'-*')
xlabel('frequency (Hz)')
ylabel('magnitude (dB)')
title('FFT of one period of the multisine realizations')

subplot(2,1,2)
plot(freq(1:nt),angle(xdft),'-*')
xlabel('frequency (Hz)')
ylabel('phase (rad)')
title('FFT of one period of the multisine realizations')

hfig(end+1) = {{gcf, "exfreq"}};

%% BLA - good way to check lines is calculated correctly

% Estimate best linear approximation, total distortion, and noise distortion
% total and noise distortion averaged over P periods and R realizations
% total distortion level includes nonlinear and noise distortion
% G: FRF; covGML: noise + NL; covGn: noise (all only on excited lines)
% if exist('fCovarFrf','file')
% Ptr = 2;
% yest = phi*y(:,Ptr:end,:,1);
% uest = permute(u(:,Ptr:end,:,1),[1,4,3,2]); % N x m x R x P
% yest = permute(yest,[1,4,3,2]); % N x p x R x P
% U = fft(uest); U = U(lines,:,:,:); % Input spectrum at excited lines
% Y = fft(yest); Y = Y(lines,:,:,:); % Output spectrum at excited lines
% [G,covGML,covGn] = fCovarFrf(U,Y);
% 
% figure;
% subplot(2,1,1); plot(freq(lines),db(abs([G(:) covGML(:) covGn(:)])));
% xlabel('Frequency (Hz)'); ylabel('Amplitude (dB)')
% legend('FRF','Total distortion','Noise distortion')
% subplot(2,1,2); plot(freq(lines),rad2deg(angle(G(:))))
% xlabel('Frequency (Hz)'); ylabel('Angle (degree)')
% 
% hfig(end+1) = {{gcf, "bla"}};
% end

%% show detect type of nonlinearity

% % output signal for one period
% % plot only half spectrum(single sided spectrum)
% dft = fft(y(:,end,r));
% nt = length(dft)/2+1;
% xdft = dft(1:nt);
% Ydot = fft(ydot(:,end,r));
% 
% figure;
% hold on
% plot(freq(ms.lines),db(xdft(ms.lines)),'.k')
% scatter(freq(ms.non_odd),db(xdft(ms.non_odd)),[])
% scatter(freq(ms.non_even),db(xdft(ms.non_even)),[])
% hold off
% 
% xlabel('frequency (Hz)')
% ylabel('magnitude (dB)')
% title('FFT of one period of the output spectrum')
% legend('excited','non\_odd','non\_even')
% 
% hfig(end+1) = {{gcf, "ms_nl_detection"}};

%% save figures
if savefig
    for i=1:length(hfig)
        h = hfig{i}{1};
        fname = hfig{i}{2};
        % change figure background from standard grey to white
        set(h, 'Color', 'w');
        figname = sprintf('./fig/b%d_A%d_%s_%s',benchmark,A,dataname,fname);
        export_fig(h, figname, '-pdf', '-png');
    end
    
end