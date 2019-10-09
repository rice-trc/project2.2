function [t,u,y] = pnlss_preparing_backbone_simulation(results, opt)
%
% prepare time data of simulated experiments for pnlss identification
%
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

ln = floor(min(cellfun(@(c) length(c), t_cell_rs))/opt.ppr)*opt.ppr;
u = permute(cell2mat(cellfun(@(c) reshape(c(1:ln),[],opt.ppr), F_cell_rs, 'UniformOutput', false)), [1, 3, 2]);
t = permute(cell2mat(cellfun(@(c) reshape(c(1:ln),[],opt.ppr), t_cell_rs, 'UniformOutput', false)), [1, 3, 2]);
t=t-t(1,:,:);
y = zeros(ln/opt.ppr, 1, number_timeframes*opt.ppr, length(DOFs));
for ii=1:length(DOFs)
    y(:,:,:,ii) = permute(cell2mat(cellfun(@(c) reshape(c(1:ln),[],opt.ppr), x_cell_rs(ii,:), 'UniformOutput', false)), [1,3,2,4]);
end
end