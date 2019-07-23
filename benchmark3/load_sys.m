function sys=load_sys(benchmark)

setup = 'New_Design_Steel';
switch benchmark
    case 1
        Dmod = [.38]*.01;
        Nmod = 1;
        nmod_setup = Nmod;
        
        thickness = .001;
         % load nonlinear coefficients (can be found e.g. analytically)
        modelname = ['data/beam_New_Design_Steel_analytical_5t_' ...
            num2str(thickness*1000) 'mm.mat'];
    case 2       
        Dmod = [.38 .12 .09 .08 .08]*.01;
        Nmod = 5;
        nmod_setup = Nmod;
        thickness = .001;
        modelname = ['data/beam_New_Design_Steel_analytical_5t_' ...
            num2str(thickness*1000) 'mm.mat'];
    case 3
        % Fundamental parameters
        Dmod = [.38 .09 .08]*.01; % first, third, fifth bending modes of flat beam
        Nmod = 3;
        % for some reason the nmod supplied to beams_for_everyone is
        % different
        nmod_setup = Nmod*2-1;
        thickness = .001;
        R=3;
        % load nonlinear coefficients (can be found e.g. via IC-method)
        modelname  = ['data/beam_msh_80_4_1_3t_steel_',...
            num2str(thickness*1000),'mm_R',num2str(R),'m.mat'];

end

%% load analytical parameters
[L,rho,E,om,PHI,~,gam] = beams_for_everyone(setup,nmod_setup,thickness);

% load nl coefficient
[p, E] = nlcoeff(modelname, Nmod, benchmark);

if benchmark == 3
    gam = gam(1:2:end);
    % om is needed for linear model, so we load the model directly
    load(modelname);
    om = model.omega;
end

% Properties of the underlying linear system
M = eye(Nmod);
D = diag(2*reshape(Dmod(1:Nmod),[],1).*om(:));
K = diag(om.^2);

% Fundamental harmonic of external forcing
Fex1 = gam;

sys = struct('M',M,'D',D,'K',K,'p',p,'E',E,'gam',gam,'L',L,'PHI',PHI,...
    'Fex1', Fex1,'rho',rho,'Nmod',Nmod);

end