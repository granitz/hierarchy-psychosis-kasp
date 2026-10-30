%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_hopf_subject_refinement.m
%
% 1. Load optimal parameters (gBest, aBest) from step 1 (or specify manually)
% 2. Load Ceff_best
% 3. For each subject, run iterative fitting with the same (gBest, aBest)
% 4. Save final subject-level connectivity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42);
maxNumCompThreads(12);

if ~exist('GRIDON','var')
    GRIDON = false;
end

% ------------- load best (g,a) from step 1 --------------
% Paths, atlas-specific filenames and the run tag all come from kasp_paths.m,
% shared with run_hopf_subject_refinement.m. Set KASP_DATA and KASP_RUN_TAG.
P = kasp_paths(ATLASNAME);
basepath     = P.basepath;
data_dir     = P.data_dir;
res_dir      = P.res_dir;
derived_data = P.derived_dir;      % this run's own folder, never data/derived_data
tspath       = P.tspath;

if GRIDON
    grpPrefix  = 'gridGroupResults';
    subjPrefix = 'subjectRefinement';
else
    grpPrefix  = 'no_grid_GroupResults';
    subjPrefix = 'no_grid_subjectRefinement';
end

% load group‐level optimum
resdir = fullfile(res_dir, 'results_group_only');
bestfile = fullfile(resdir, ...
    sprintf('%s_group-%s_tau-%d_med-%d_atlas-%s.mat', ...
        grpPrefix, GROUP, Tau, MED, ATLASNAME));

bestdat   = load(bestfile);
gBest     = bestdat.gBest;
aBest     = bestdat.aBest;
Ceff_best = bestdat.Ceff_best; % group-level connectivity for optimal solution

%%%
% --------------- Define parameters ------------
N        = str2double(ATLASNAME);
TR       = 2.000;  % Repetition time
sigma    = 0.01;   % noise amplitude
epsFC    = 0.0004;
epsFCtau = 0.0001;
maxC     = 0.2;
max_iters= 5000;

% Load SC
load(P.sc_file);
if size(SC, 2) ~= N
    error('SC in %s is %dx%d but ATLASNAME says %d parcels.', P.sc_file, size(SC,2), size(SC,3), N);
end

SC = squeeze(mean(SC,1));
SC = SC / max(SC(:)) * maxC;

ignoreFile = P.ignore_file;
load(ignoreFile, 'IgnoreNodes');
IgnoreNodes = double(IgnoreNodes(:))';
indexN = setdiff(1:N, IgnoreNodes);
fprintf('IgnoreNodes: %d dropped, %d retained\n', numel(IgnoreNodes), numel(indexN));

indexN = setdiff(1:N, IgnoreNodes);
SC     = SC(indexN,indexN);

disp('Loaded structural connectivity')

% Load subjects name
% The participant exclusions come from the SAME file as the node list
% (IgnoreNodes.mat, field ExcludeSubjects), so the two cannot diverge.
[subjects, subinfo] = load_subject_list(basepath, GROUP, MED, ignoreFile);

% Load timeseries

NSUB = length(subjects);
TS = [];  % Will be NSUB x N x T

for iSub = 1:NSUB
    subID = subjects{iSub};

    fname = fullfile(tspath, [subID P.ts_suffix]);
    
    if isfile(fname)
        data = load(fname);
        % Get the timeseries variable (node x time)

        ts_sub = data.ts;  % node x time

        % Keep only valid nodes: indexN x time
        ts_sub = ts_sub(indexN, :);
        % Reshape to 1 x N x T and concatenate
        ts_sub = reshape(ts_sub, 1, length(indexN), size(ts_sub, 2));
        TS = cat(1, TS, ts_sub);
    else
        fprintf('Warning: File not found for %s\n', subID);
    end
end

if size(TS,1) ~= NSUB
    error('TS has %d subjects but the list has %d — a file is missing.', ...
          size(TS,1), NSUB);
end

% Subject-level refinement cannot tolerate missing nodes. Unlike the group fit,
% which averages across participants with nanmean, this fits each participant
% individually: a NaN node makes Cov_sub(i,i) NaN, which makes sigratio NaN,
% which writes NaN into Cnew_sub through the gradient update -- and the rescale
% `Cnew_sub / max(Cnew_sub(:))` then turns the WHOLE matrix NaN, because max()
% of an array containing NaN is NaN. One missing node silently destroys that
% participant's fit. Fail here instead.
nanmask = squeeze(any(isnan(TS), 3));                  % NSUB x Nvalid
if any(nanmask(:))
    [si, ni] = find(nanmask);
    % Reports a position, not an identifier: this message ends up in logs.
    error('hopf:missingNodes', ...
          ['%d node-subject pairs are missing (e.g. participant %d of %d at node %d).\n' ...
           'Regenerate IgnoreNodes.mat with make_ignore_nodes.py using the ' ...
           'any-subject rule, listing the participants to exclude.'], ...
          numel(si), si(1), NSUB, indexN(ni(1)));
end
fprintf('no missing nodes: %d subjects x %d nodes complete\n', NSUB, numel(indexN));

% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                  % lowpass frequency of filter (Hz)
fhi = 0.08;                   % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

% Compute frequencies from subject timeseries
Nvalid = length(indexN);
f_diff_sub = zeros(NSUB, Nvalid);
FCemp = zeros(NSUB, Nvalid, Nvalid);

for sub_i = 1:NSUB
    sub_i
    clear signal_filt Power_Areas;
    % Extract timeseries for this subject (N x T)
    ts = squeeze(TS(sub_i, :, :));

    % Preallocate: building by assignment mis-shapes the array if seed 1 is
    % a node with no data. (Unreachable given the check above, but kept so a
    % relaxed criterion fails visibly rather than silently.)
    signal_filt = nan(Nvalid, size(ts,2));

    for seed=1:Nvalid
        if all(isnan(ts(seed,:)))
            continue                      % filtfilt rejects non-finite input
        end
        ts(seed,:)=detrend(ts(seed,:)-mean(ts(seed,:)));
        signal_filt(seed,:) = filtfilt(bfilt,afilt,ts(seed,:));
    end
    signal_filt=signal_filt(:,10:end-10);

    [Ns, Tmaxred] = size(signal_filt);
    TT = Tmaxred;
    Ts = TT * TR;  % Total sampling time
    freq = (0:TT/2-1) / Ts;  % Frequency range
    nfreqs = length(freq);

    % Perform spectral analysis on filtered data
    for seed = 1:Nvalid
        if all(isnan(signal_filt(seed, :)))
            % max() on an all-NaN spectrum returns index 1, which would assign
            % freq(1) = 0 -- a silently wrong intrinsic frequency.
            f_diff_sub(sub_i, seed) = NaN;
            continue
        end
        pw = abs(fft(zscore(signal_filt(seed, :))));
        PowSpect = pw(1:floor(TT/2)).^2 / (TT/TR);
        Power_Areas = gaussfilt(freq, PowSpect, 0.005);
        [~, index] = max(Power_Areas);
        index = squeeze(index);
        f_diff_sub(sub_i, seed) = freq(index);
    end
end

% Average frequencies across subjects
f_diff = mean(f_diff_sub, 1, 'omitnan');
if any(~isfinite(f_diff))
    error('f_diff is non-finite at %d node(s).', sum(~isfinite(f_diff)));
end
wo = 2*pi * f_diff(:);

disp('Computed frequencies and FC from subject timeseries')


% aVec
aVec = aBest * ones(Nvalid,1);

% lags
lags_model = [TR*Tau];

% ---------- subject-level refinement -----------
Ceff_Subj      = zeros(NSUB, Nvalid, Nvalid);
fittFC_subj    = zeros(NSUB,1);
fittCOVtau_subj= zeros(NSUB,1);
FCsim_Subj     = zeros(NSUB, Nvalid, Nvalid);
CovTau_sim_Subj= zeros(NSUB, Nvalid, Nvalid);

for nsub = 1:NSUB
    disp(['Refining subject ' num2str(nsub) ' ...']);

    ts = TS(nsub,:,:);
    ts = squeeze(ts);
    ts = ts(:, 10:end-10);

    % Empirical FC
    FCemp_sub   = corrcoef(ts');
    Cov_sub     = cov(ts');
    CovTau_emp  = zeros(Nvalid,Nvalid);
    sigratio    = zeros(Nvalid,Nvalid);
    for i=1:Nvalid
        for j=1:Nvalid
            sigratio(i,j) = 1/sqrt(Cov_sub(i,i))/sqrt(Cov_sub(j,j));
            [clag, lags]  = xcov(ts(i,:), ts(j,:), Tau);
            idx = find(lags==Tau);
            CovTau_emp(i,j) = clag(idx)/size(ts,2);
        end
    end
    CovTau_emp = CovTau_emp .* sigratio;

    % FCemp_sub(isnan(FCemp_sub)) = 0;
    % CovTau_emp(isnan(CovTau_emp)) = 0;

    % Start from optimal group-level Ceff
    Cnew_sub = Ceff_best;
    olderror_sub = Inf;

    for iterS=1:max_iters
        [FCsim_sub, Cov0_sub, Ct_sub] = HopfModel_LNA(...
            Cnew_sub, aVec, gBest, wo, sigma, lags_model);

        % Cov at TR
        CovTau_sim0 = squeeze(Ct_sub);

        sigratiosim_sub = zeros(Nvalid,Nvalid);
        for i=1:Nvalid
            for j=1:Nvalid
                sigratiosim_sub(i,j) = 1/sqrt(Cov0_sub(i,i))/sqrt(Cov0_sub(j,j));
            end
        end
        CovTau_sim = CovTau_sim0 .* sigratiosim_sub;

        % errors
        errFC_s  = mean( (FCemp_sub(:)     - FCsim_sub(:)).^2 );
        errCOV_s = mean( (CovTau_emp(:)    - CovTau_sim(:)).^2 );
        totErr_s = errFC_s + errCOV_s;

        if mod(iterS,100)<0.1
            iterS
            if (olderror_sub - totErr_s)/totErr_s < 1e-3
                disp('No improvement => stop.');
                break;
            end
            if totErr_s > olderror_sub
                disp('Error up => stop.');
                break;
            end
            olderror_sub = totErr_s;
        end

        % gradient update
        for i=1:Nvalid
            for j=1:Nvalid
                if SC(i,j)>0 % only update existing connections
                    Cnew_sub(i,j) = Cnew_sub(i,j) ...
                      + epsFC   * (FCemp_sub(i,j)  - FCsim_sub(i,j)) ...
                      + epsFCtau* (CovTau_emp(i,j) - CovTau_sim(i,j));
                    if Cnew_sub(i,j)<0
                        Cnew_sub(i,j)=0;
                    end
                end
            end
        end
        % rescale
        Cnew_sub = Cnew_sub / max(Cnew_sub(:)) * maxC;
    end

    Ceff_Subj(nsub,:,:)      = Cnew_sub;
    FCsim_Subj(nsub,:,:)      = FCsim_sub;
    CovTau_sim_Subj(nsub,:,:) = CovTau_sim;

    % final correlations
    fc_ = corr2(FCemp_sub(tril(true(Nvalid), -1)), ...
                FCsim_sub(tril(true(Nvalid), -1)));
    fittFC_subj(nsub) = fc_;
    cv_ = corr2(CovTau_emp(tril(true(Nvalid), -1)), ...
                CovTau_sim(tril(true(Nvalid), -1)));
    fittCOVtau_subj(nsub) = cv_;
end

% Expand Ceff_Subj to full size (NSUB x 416 x 416) with NaN for ignored nodes
Ceff_Subj_full = nan(NSUB, N, N);
for nsub = 1:NSUB
    Ceff_Subj_full(nsub, indexN, indexN) = squeeze(Ceff_Subj(nsub, :, :));
end

% Expand FCsim_Subj and CovTau_sim_Subj to full size (NSUB x 416 x 416) with NaN for ignored nodes
FCsim_Subj_full = nan(NSUB, N, N);
CovTau_sim_Subj_full = nan(NSUB, N, N);
for nsub = 1:NSUB
    FCsim_Subj_full(nsub, indexN, indexN) = squeeze(FCsim_Subj(nsub, :, :));
    CovTau_sim_Subj_full(nsub, indexN, indexN) = squeeze(CovTau_sim_Subj(nsub, :, :));
end

% save subject‐level results with matching prefix
resdir2 = fullfile(res_dir, 'results_subject_refinement');
if ~exist(resdir2, 'dir')
    mkdir(resdir2);
end

%% Hierarchy Ceff

%% PB
hierarchicallevels_full = nan(NSUB, N);  % Full N=416 array with NaN

for nsub=1:NSUB
    Ceff=squeeze(Ceff_Subj(nsub,:,:));
    A=Ceff';
    d=sum(A)';
    delta=sum(A,2);
    u=d+delta;
    v=d-delta;
    Lambda=diag(u)-A-A';
    Lambda(1,1)=0;
    gamma=linsolve(Lambda,v);
    gamma=gamma-min(gamma);
    hierarchicallevels(nsub,:)=gamma';

    % Expand gamma to full size (N=416) with NaN for ignored nodes
    hierarchicallevels_full(nsub, indexN) = gamma';

    H=(meshgrid(gamma)-meshgrid(gamma)'-1).^2;
    F0=sum(sum((A.*H)))/sum(sum(A));
    trophiccoherence(nsub)=1-F0;
end

outname = sprintf('%s_group-%s_tau-%d_med-%d_atlas-%s.mat', ...
    subjPrefix, GROUP, Tau, MED, ATLASNAME);

code_commit = P.code_commit;      % provenance: the code that produced this file
code_dirty  = P.code_dirty;
run_tag     = P.run_tag;
save(fullfile(resdir2, outname), ...
     'Ceff_Subj_full','hierarchicallevels_full','trophiccoherence', 'fittFC_subj', 'fittCOVtau_subj', ...
     'FCsim_Subj_full','CovTau_sim_Subj_full', 'IgnoreNodes', 'indexN','subjects', 'subinfo', ...
     'gBest', 'aBest', 'Tau', 'sigma','MED','subjects','indexN', ...
     'code_commit', 'code_dirty', 'run_tag');
