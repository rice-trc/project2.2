function [p, E] = nlcoeff(fname,Nmod,benchmark)
% load nonlinear coefficients (can be found e.g. analytically)

% backwards compability
if nargin < 3
    benchmark = 3;
end

if benchmark < 3
    model = load(fname);
else
    load(fname);
end

% polynomial terms
p_quad = zeros(sum(1:Nmod),Nmod);
p_cub = zeros(sum(cumsum(1:Nmod)),Nmod);
ctr_quad = 1; ctr_cub = 1;

for jj = 1:Nmod
    for kk = jj:Nmod
        % quadratic terms
        p_quad(ctr_quad,jj) = p_quad(ctr_quad,jj)+1;
        p_quad(ctr_quad,kk) = p_quad(ctr_quad,kk)+1;
        ctr_quad = ctr_quad+1;
        for ll = kk:Nmod
            % cubic terms
            p_cub(ctr_cub,jj) = p_cub(ctr_cub,jj)+1;
            p_cub(ctr_cub,kk) = p_cub(ctr_cub,kk)+1;
            p_cub(ctr_cub,ll) = p_cub(ctr_cub,ll)+1;
            ctr_cub = ctr_cub+1;
        end
    end
end

switch(benchmark)
    case {1,2}
        p = [p_cub];
    case 3
        p = [p_quad; p_cub];
end

% coefficients
E=zeros(sum(cumsum(1:Nmod)),Nmod);

for rr = 1:Nmod
    ctr = 1;
    if benchmark == 3
        for jj = 1:Nmod
            for kk = jj:Nmod
                % quadratic coeffs
                E(ctr,rr) = model.a(jj,kk,rr);
                ctr = ctr+1;
            end
        end
    end
    for jj = 1:Nmod
        for kk = jj:Nmod
            for ll = kk:Nmod
                % cubic coeffs
                E(ctr,rr) = model.b(jj,kk,ll,rr);
                ctr = ctr+1;
            end
        end
    end
end

end