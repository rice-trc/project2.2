function [time, y, z] = RK_GEN_TV(func, tvec, IC, ICz, pars)
%RK_GEN_TV integrates generic function with static evolving parameter using
%fixed step explicit RK schemes
%of the form,
%   [y', z] = func(t, y, z)
% USAGE:
%   [time, y, z] = RK_GEN_tv(func, tvec, IC, ICz, pars);
% INPUTS:
%   func    : function handle as above
%   tvec    : Ntx1 vector of time points required in the output
%   IC      : Initial conditions in the dynamic states
%   ICs     : Initial conditions in the static states
%   pars    : structure with parameters
%       a       : Butcher table a
%       b       : Butcher table b
%       c       : Butcher table c
% OUTPUTS:
%   time    : Ntx1 time vector. These are the closest ones to tvec but not
%               exactly the same.
%   y       : NtxNdof State vector time series. Note that this is on the
%               time grid "time", which is not exactly "tvec". For 
%               obtaining at "tvec", one will have to further conduct
%               interpolations.
%   z       : NtxNz Static state vector time series

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % RK 45 Butcher Tableau
% % pars.a = [0 0 0 0 0 0; 
% %           1/4 0 0 0 0 0; 
% %           3/32 9/32 0 0 0 0; 
% %           1932/2197 -7200/2197 7296/2197 0 0 0;
% %           439/216 -8 3680/513 -845/4104 0 0;
% %          -8/27 2 -3544/2565 1859/4104 -11/40 0];
% % pars.b = [16/135 0 6656/12825 28561/56430 -9/50 2/55];
% % pars.c = [0 1/4 3/8 12/13 1 1/2];
% % % Display
% % pars.Display = 'on';
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


    a = pars.a;
    b = pars.b;
    c = pars.c;
    
    ord = length(c);

    dt = tvec(2)-tvec(1);
    Np = length(tvec);
    if Np==2
        error('You have to input the required time vector for this function');
    end
    time = tvec;
    y = zeros(Np, length(IC));
    y(1, :) = IC;
    z = zeros(Np, length(ICz));
    z(1, :) = ICz;
    
    k  = zeros(ord, length(IC));
    kz = zeros(ord, length(ICz));
    n  = 1;
    for n=1:Np-1
        dt = time(n+1)-time(n);
        k = zeros(size(k));
        for i=1:ord
            % Option 1: Do not update z between RK steps
            % [k(i,:), kz(i,:)] = func(time(n)+c(i)*dt, (y(n,:)+dt*a(i,:)*k)', z(n,:)');
            
            % Option 2: Update z consistently (Better)
            [k(i,:), kz(i,:)] = func(time(n)+c(i)*dt,...
                (y(n,:)+dt*a(i,:)*k)', (z(n,:)+a(i,:)*(kz-z(n,:)))');
        end
        
        % UPDATE
        y(n+1,:) = y(n,:) + dt*b*k;

        % z(n+1, :) = kz(end,:);  % Update z as per last step
        z(n+1,:) = z(n,:) + b*(kz-z(n,:));  % Update z consistently (Better)

        n = n+1;
        if strcmp(pars.Display,'on')
            fprintf("t=%e/%e n=%d/%d\n", time(n), tvec(end), n, length(tvec));
        end
    end
    time(n:end) = [];
    y(n:end,:) = [];
    z(n:end,:) = [];
end