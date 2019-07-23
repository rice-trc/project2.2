% produce plot where the multisine amplitude increases linear from A1 to A2
% during on period and then repeat for P periods.
% This might be usefull for systems with clearence, where the system is
% complely linear at low amplitude but have impacts at high amplitude

clc
close all


A1 = 1;
A2 = 15;

N = 1e3;
Nt = 2^14;

f1 = 10;
f2 = 100;
f0 = (f2-f1)/N;
fs = f0*Nt;
T = 1/f0;
P = 4;

U = (A2-A1)/T;
A = 1;

r = 1;
[fex, MS{r}] = multisine(f1, f2, N, A, [], [], r);

t = linspace(0,P*T,P*Nt+1); t(end) = [];
u = fex(t);

% hacky way of producing, for P = 2;
% u2 = u .* [U*t(t<T)+1, U*t(t>=(P-1)*T & t<P*T)+1+(P-1)*(A1-A2)];
ampstr = '@(t,U,A1,A2,T) [';
for i=1:P
    ampstr = [ampstr sprintf(' U*t(t>=(%d-1)*T & t<%d*T)+1+(%d-1)*(A1-A2)',i,i,i)];
end
ampstr = [ampstr ']'];
amptmp = str2func(ampstr);
amp = @(t) amptmp(t,U,A1,A2,T);

u2 = u .* amp(t);

figure(1), clf, hold on
plot(t,u,'--')
plot(t,u2,'Color',[1,0,0,0.2])
xlabel('time (s)')
ylabel('multisine')
legend('original','scaled')

fpath = './FIGURES/pnlss/';
% export_fig(gcf,sprintf('%s./arrow_full',fpath),'-png')
%% zoomed

indexOfInterest = (t < T*1.002 ) & (t > T*0.998);
figure(2), clf, hold on
plot(t(indexOfInterest),u(indexOfInterest))
plot(t(indexOfInterest),u2(indexOfInterest),'--')
if P>1
    plot(t(Nt+1),u(Nt+1),'rx','MarkerSize',12)
end

ylim([-1.5 1.5])
xlabel('time (s)')
ylabel('multisine')
legend('original','scaled')

% export_fig(gcf,sprintf('%s./arrow_zoom',fpath),'-png')

%% frequency
figure(3),clf, hold on
U1 = fft(u(1:Nt));
U2 = fft(u2(1:Nt));
ndft = length(U2)/2;
freq = (0:Nt-1)/Nt*fs;
plot(freq(1:ndft),db(abs(U1(1:ndft))))
plot(freq(1:ndft),db(abs(U2(1:ndft)))) %,'Color',[1,0,0,0.2])
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
legend('original','scaled')
% export_fig(gcf,sprintf('%s./arrow_freq',fpath),'-png')