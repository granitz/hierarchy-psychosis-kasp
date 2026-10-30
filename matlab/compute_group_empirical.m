function [FCemp_group, COVtauemp_group, FCPB, COVtauPB] = compute_group_empirical(TS, Tau, indexN)
% TS shape: [NSUB, N, TimePoints]
% Tau is the lag in "time points" (assuming 1 time point = 1 TR)
% Returns group-average FC(0) and Cov(tau).

NSUB = size(TS,1);
N    = size(indexN,2);
FCPB    = zeros(NSUB,N,N);
COVtauPB = zeros(NSUB,N,N);

for sub_i = 1:NSUB
    sub_i
    ts = squeeze(TS(sub_i,:,:)); % time series for subject sub_i

    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
    end

    ts = ts(:,10:end-10); % if you want to remove edges

    % Empirical FC(0)
    FC0 = corrcoef(ts');
    FCPB(sub_i,:,:) = FC0;

    % Cov(tau)
    Cov_ = cov(ts');
    sigratio = zeros(N,N);
    Cov_tau = zeros(N,N);
    for i=1:N
        for j=1:N
            sigratio(i,j) = 1/sqrt(Cov_(i,i))/sqrt(Cov_(j,j));
            [clag, lags]  = xcov(ts(i,:), ts(j,:), Tau);
            idx = find(lags==Tau);
            Cov_tau(i,j) = clag(idx)/size(ts,2); % or adjust for 'coeff'
        end
    end
    Cov_tau = Cov_tau .* sigratio;
    COVtauPB(sub_i,:,:) = Cov_tau;
end

% group-average
FCemp_group     = squeeze(nanmean(FCPB,1));
COVtauemp_group = squeeze(nanmean(COVtauPB,1));

