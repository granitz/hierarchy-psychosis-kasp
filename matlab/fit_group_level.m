function [Ceff_group, errFC, errCOVtau, finalError, FCsim, COVtauSim] = fit_group_level(...
    Cinit, aVec, gVal, wo, sigma, ...
    FCemp_group, COVtauemp_group, ...
    epsFC, epsFCtau, max_iters, maxC, lags_model)
%
% Runs the iterative gradient-based update for group-level connectivity.
% Returns the final connectivity matrix and the final total error.

Cnew     = Cinit;
N        = size(Cinit,1);
olderror = Inf;
errorNow = Inf;
for iter = 1:max_iters
    % Call LNA
    [FCsim, Cov0, Ct] = HopfModel_LNA(Cnew, aVec, gVal, wo, sigma, lags_model);
    COVtauSim0 = squeeze(Ct);

    sigratiosim = zeros(N,N);
    for i=1:N
        for j=1:N
            sigratiosim(i,j) = 1/sqrt(Cov0(i,i))/sqrt(Cov0(j,j));
        end
    end
    COVtauSim = COVtauSim0 .* sigratiosim;

    % errors
    errFC(iter)     = mean( (FCemp_group(:) - FCsim(:)).^2 );
    errCOVtau(iter) = mean( (COVtauemp_group(:) - COVtauSim(:)).^2 );
    errorNow        = errFC(iter) + errCOVtau(iter);

    % Stopping conditions
    if mod(iter,100)<0.1
        iter
        if (olderror - errorNow)/errorNow < 1e-3
            break;
        end
        if errorNow > olderror
            break;
        end
        olderror = errorNow;
    end

    % gradient update
    for i=1:N
        for j=1:N
            if Cinit(i,j)>0 % only update existing connections
                Cnew(i,j) = Cnew(i,j) ...
                  + epsFC   * (FCemp_group(i,j)     - FCsim(i,j)) ...
                  + epsFCtau* (COVtauemp_group(i,j) - COVtauSim(i,j));
                if Cnew(i,j)<0
                    Cnew(i,j)=0;
                end
            end
        end
    end
    % rescale
    Cnew = Cnew / max(Cnew(:)) * maxC;
end

Ceff_group = Cnew;
finalError = errorNow;
