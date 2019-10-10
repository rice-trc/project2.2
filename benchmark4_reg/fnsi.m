clear variables

addpath('~/ownCloud/phd/code/matlab/fnsi')
load('data/pol_A200_upsamp1_fs700_eps01.mat')

nldof = 3;
cr_true = 1e9;
i = 40;
order = 6;

% convert from python to matlab format
lines = double(lines + 1);
fs = double(fs);
u = permute(u, [1,4,3,2]);
y = permute(y, [1,4,3,2]);

[N, P, R, p] = size(y);
[N, P, R, m] = size(u);

% Last realization, last period for performance testing
utest = u(:,end,R,:); utest = reshape(utest,[],m);
ytest = y(:,end,R,:); ytest = reshape(ytest,[],p);

% One but last realization, last period for validation and model selection
uval = u(:,end,R-1,:); uval = reshape(uval,[],m);
yval = y(:,end,R-1,:); yval = reshape(yval,[],p);

% All other repeats for estimation\
Ptr = 2;
Rest = 1;
uest = u(:,Ptr+1:end,Rest,:);% uest = reshape(uest,[],m);
yest = y(:,Ptr+1:end,Rest,:);% yest = reshape(yest,[],p);

uest = permute(uest,[4,1,2,3]); % m x N x P
yest = permute(yest,[4,1,2,3]); % m x N x P

%[m,cu] = size(uest);if (m > cu), uest = uest';[m,~] = size(uest);end
%[p,cy] = size(yest);if (p > cy), yest = yest';[p,~] = size(yest);end

%%
%Frequency axis
freq = (0:N-1)*fs/N;

U = fft(uest,[],2)/sqrt(N);
Y = fft(yest,[],2)/sqrt(N);

[Umean,~] = FNSI_MeanVar(U);
[Ymean,WY] = FNSI_MeanVar(Y);

%Nonlinearity definition
ynl = sum(yest,3)/(P-Ptr);

%nl = abs(ynl(1,:)).^3.*sign(ynl(1,:));
nl = ynl(nldof,:).^3;


[nnl,cnl] = size(nl);if (nnl > cnl);nl = nl';[nnl,~] = size(nl);end

scaling = zeros(nnl,1);
for j = 1:nnl,
    scaling(j) = std(uest(1,:))/std(nl(j,:));
    nl(j,:) = scaling(j)*nl(j,:);
end %j

NL = fft(nl,[],2)/sqrt(N);

% % Stabilisation diagram
% 
% nlist = 2:2:20;
% SDin = FNSI_SD(E,Ymean,[],i,nlist,flines,N,fs);
% 
% tol.freq = 1;
% tol.ep = 5;
% tol.mode = 0.98;
% tol.mac = 'complex';
% display.unstab = 'x';
% 
% SDout = FNSI_PlotSD(SDin,nlist,tol,display,0);
% xlim([0 1]);ylim([0 max(nlist)+2]);

%FNSI analysis
E = [Umean ; -NL];

bd_method = 'explicit'; % or 'optim', 'nr'
%bd_method = 'optim';
bd_method = 'nr';
[ASUB,BSUB,CSUB,DSUB,Ad,Bd] = FNSI(E,Ymean,[],i,order,lines,N,fs, bd_method);

%Output noise weighting matrix W can be considered if noisy time series are processed
% [ASUB,BSUB,CSUB,DSUB,Ad,Bd] = FNSI(E,Ymean,WY,i,order,lines,N,fs, bd_method);

%Conversion from state to physical space
inl = [3 0];
iu = 1;
iFRF = 1;

[knl,HSUB] = FNSI_State2Physical2(fs,N,lines,ASUB,BSUB,CSUB,DSUB,inl,iu,iFRF,scaling);

lpid = FNSI_LinearParameters(ASUB,CSUB);

%disp('Natural frequency and damping ratio (%)');
%disp(lpid.natfreq);
%disp(lpid.ep);

figure;plot(freq(lines+1), real(knl(:,1)));%xlim([0 1])%;ylim([0.4 0.6])
xlabel('Frequency (Hz)');ylabel('Real part of the NL coefficient (N/m^3)')

c = mean(real(knl),1);
cim = mean(imag(knl),1);
logr = log10(abs(c./cim));

%disp(' ');
disp('Error on the nonlinear coefficient (%)');
disp(100*(c-cr_true)/cr_true);

%disp(' ');
disp('Ratio of the real and imaginary parts of the nonlinear coefficient (log)');
disp(logr)

