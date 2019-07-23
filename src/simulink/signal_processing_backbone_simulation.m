function [res] = signal_processing_backbone_simulation(results, opt)
%
% Calculate linear modal characteristics of simulated experiments
%
% created 05.04.2019 M. Scheel
%
%  editing history
%  index	date		who		comment
%  ------------------------------------------------------------------------
% [001]     
%  ------------------------------------------------------------------------

% -------------------------------------------------------------------------
%%%%%%%%%%% Loading data
% -------------------------------------------------------------------------

tvals = results.tvals; % time vector
xvals = results.disp; % displacement signals
Fvals =  results.Fvals; % force signal
freqvals = results.freqvals; % frequency signal

% -------------------------------------------------------------------------
%%%%%%%%%%% Options for analysis
% -------------------------------------------------------------------------
DOFs = 1:size(xvals,2); %vector with DOFs for evaluation
exc_DOF = opt.eval_DOF;

%options for quality indicator
n_harm = opt.n_harm; %number of harmonics considered
min_harm_level = opt.min_harm_level; %minimal level relative to highest peak

periods = opt.periods;
Fs = opt.Fs;


% -------------------------------------------------------------------------
%%%%%%%%%%% Extracting time frames to be evaluated
% -------------------------------------------------------------------------

time = results.Signalbuilder;
timeframes_end = time(3:2:end);
if size(timeframes_end,1) == 1
    timeframes_end = timeframes_end';
end
if size(time,1) == 1
    time = time';
end
timeframes = zeros(length(timeframes_end),2);
timeframes(:,2) = timeframes_end;
timeframes_diff = timeframes_end - time(2:2:end-1);
timeframes(:,1) = timeframes(:,2) - timeframes_diff/8;
timeframes(timeframes(:,2)>tvals(end),:)=[];
number_timeframes = size(timeframes,1);

% extracting excitation frequency of each time frame
frequencies = zeros(1,number_timeframes);
freq_variance = zeros(1,number_timeframes);
for ii = 1:number_timeframes
    t_orig = tvals(tvals>timeframes(ii,1));
    freq_orig = freqvals(tvals>timeframes(ii,1));
    freq_orig = freq_orig(t_orig<timeframes(ii,2));
    freq_variance(ii) = max(freq_orig)-min(freq_orig);
    frequencies(ii) = mean(freq_orig);
end

periodtimes = 1./frequencies;
timeframes(:,1) = timeframes(:,2) - periods * periodtimes';

% -------------------------------------------------------------------------
%%%%%%%%%%% Computation of the FFT of the signals
% -------------------------------------------------------------------------

x_cell_rs = cell(length(DOFs),number_timeframes);
x_cell = cell(length(DOFs),number_timeframes);
X_cell = cell(length(DOFs),number_timeframes);
   

%loop over DOFs
for mm = 1:length(DOFs)

    t_cell_rs = cell(1,number_timeframes);
    t_cell = cell(1,number_timeframes);
    F_cell_rs = cell(1,number_timeframes);
    F_cellt = cell(1,number_timeframes);

    %loop for extraction of time frames
    for ii = 1:number_timeframes
        t_orig = tvals(tvals>timeframes(ii,1));
        x_orig = xvals(tvals>timeframes(ii,1),mm);
        F_orig = Fvals(tvals>timeframes(ii,1));
        x_orig = x_orig(t_orig<timeframes(ii,2));
        F_orig = F_orig(t_orig<timeframes(ii,2));
        t_orig = t_orig(t_orig<timeframes(ii,2));
        %save signal frames to cells

        t_cell{ii} = t_orig;
        x_cell{mm,ii} = x_orig;
        F_cellt{ii} = F_orig;  
    end
    
    clear t_orig x_orig F_orig
  

%%%% this part is needed if variable step size simulation results must be
%%%% interpolated

    if opt.var_step == 1

        for jj = 1:size(t_cell,2)
            %retrieve signals from cells
            t_orig = t_cell{jj};
            x_orig = x_cell{mm,jj};
            F_orig = F_cellt{jj};
            %calculate desired time vector and interpolate x        
            t_des = min(t_orig):1/Fs:max(t_orig);        
            x_des = interp1(t_orig,x_orig,t_des)';
            F_des = interp1(t_orig,F_orig,t_des)';
            %save resampled signal frames to cells
            t_cell_rs{jj} = t_des;
            x_cell_rs{mm,jj} = x_des;
            F_cell_rs{jj} = F_des;

        end
    end

    F_cell = cell(1,number_timeframes);
    f_cell = cell(1,number_timeframes);
    %calculation of FFTs of (interpolated) signals
    % loop over number of timeframes
    for kk = 1:number_timeframes
        t = t_cell_rs{kk};
        x = x_cell_rs{mm,kk};
        F = F_cell_rs{kk};
        L = length(t);
        X = fft(x,L)/L;
        F = fft(F,L)/L;
        f = Fs/2*linspace(0,1,L/2+1);

        X_cell{mm,kk} = [X(1);2*X(2:floor(length(X)/2)+1)];
        f_cell{kk} = f;
        F_cell{kk} = [F(1);2*F(2:floor(length(X)/2)+1)];
    end
end % loop over DOFS

clearvars xvals tvals Fvals t_cell_rs x_cell_rs F_cell_rs

% -------------------------------------------------------------------------
%%%%%%%%%%% Postprocessing
% -------------------------------------------------------------------------

P_act_1 = zeros(1,number_timeframes);
Forces = zeros(1,number_timeframes);
Psi_tildes = zeros(length(DOFs),number_timeframes);
Force_1 = zeros(1,number_timeframes);
frequencies = zeros(1,number_timeframes);
PBMIF = zeros(1,number_timeframes);
Disps = zeros(1,number_timeframes);
phase_all_i = zeros(length(DOFs),number_timeframes);
X_harm = cell(1,number_timeframes);
F_harm = cell(1,number_timeframes);

% -------------------------------------------------------------------------
%%%%%%%%%%% Finding resonant frequencies
% -------------------------------------------------------------------------

for ll = 1:number_timeframes
  
    [~,X_LOCS] = findpeaks(abs(X_cell{exc_DOF,ll}),'MINPEAKHEIGHT',max(abs(X_cell{exc_DOF,ll}))*min_harm_level,'SORTSTR','descend','NPEAKS',n_harm);
	freq_X_PKS = f_cell{ll}(X_LOCS);     
	% X_LOCS are peak location as index, freq_X_PKS are frequencies in HZ 
    % at peaks
        
	%filter out spourios peaks in constant part
    spurious_PKS = find(freq_X_PKS<5);
    freq_X_PKS(spurious_PKS) = [];
    X_LOCS(spurious_PKS) = [];

    % values of the variables where X has peaks
    % each value is equal to one harmonic
    F_harm{ll} = F_cell{ll}(X_LOCS);
    X_harm{ll} = zeros(length(DOFs),length(X_LOCS));
    for ii = 1:length(DOFs)
        X_harm{ll}(ii,:) = X_cell{ii,ll}(X_LOCS);
    end
    
% -------------------------------------------------------------------------
%%%%%%%%%%% Computing phase difference between force and displacement
% -------------------------------------------------------------------------

    phase_F = atan2(real(F_harm{ll}),imag(F_harm{ll}));
    for ii = 1:length(DOFs)
        phase_tmp = atan2(real(X_cell{ii,ll}(X_LOCS)),imag(X_cell{ii,ll}(X_LOCS)));
        phase_tmp=(phase_tmp(1)-phase_F(1))*180/pi;
        neg_phase = find(phase_tmp < 0);
        phase_tmp(neg_phase) = phase_tmp(neg_phase)+360; 
        phase_all_i(ii,ll) = phase_tmp; 
    end
    % phase between 0° and 180° (at resonance 90°)
    
%----------------------------------------------------------------------
% calculate Power Based Nonlinear Mode Indicator Function by Peter
%----------------------------------------------------------------------
    V_harm = zeros(length(X_LOCS),1);
    for ii = 1:length(V_harm)
        V_harm(ii) = (2*pi*1i*f_cell{ll}(X_LOCS(ii)))*(X_cell{exc_DOF,ll}(X_LOCS(ii)));
    end

    P_act_harm = 0.5* abs(F_harm{ll}).*abs(V_harm).*sind(phase_all_i(exc_DOF));
    S = 0.5*sqrt(sum((abs(F_harm{ll})).^2)*sum(abs(V_harm)).^2); %apparent power

    PBMIF(ll) = abs(sum(P_act_harm))/S;

%----------------------------------------------------------------------
% store results in vectors over timeframes 
%----------------------------------------------------------------------

    frequencies(ll) = freq_X_PKS(1); % resonant frequencies in Hz (only first harmonic) 

    Forces(ll) = sqrt(sum((abs(F_harm{ll})).^2/2));
    Disps(ll) = sqrt(sum((abs(X_harm{ll}(exc_DOF,:))).^2/2));

    P_act_1(ll) =  P_act_harm(1);
    Force_1(ll) = (F_cell{ll}(X_LOCS(1)))*exp(1i*phase_F(1));
    Psi_tildes(:,ll) = X_harm{ll}(:,1).*exp(1i*phase_F(1)); % has same phase as force
 
end

%----------------------------------------------------------------------
% store results
%----------------------------------------------------------------------

res.F_i = Force_1; % force of the first harmonic with phase of force
res.om_i = frequencies * 2 * pi; % nonlinear eigenfrequency
res.Psi_tilde_i = Psi_tildes; % nonlinear eigenvectors (not normalized)
res.P_act_1 = P_act_1; % active power of the first harmonic
res.PBMIF = PBMIF; % Power Based Mode Indicator Function (Peter)
% phase difference phase_X-phase_F in deg
res.phases = phase_all_i;
res.options = opt;

k_i = sqrt(((abs(Forces)).^2-(1/sqrt(2)*abs(Force_1)).^2))./(abs(Forces));
% fundamental harmonic content force
res.gamma_i = sqrt(1-k_i.^2);

k_x_i = sqrt(((abs(Disps)).^2-(1/sqrt(2)*abs(res.Psi_tilde_i(exc_DOF,:))).^2))./(abs(Disps));
% fundamental harmonic content displacement
res.gamma_x_i = sqrt(1-k_x_i.^2);

