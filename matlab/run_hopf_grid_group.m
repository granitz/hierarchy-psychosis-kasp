%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run_hopf_grid_group.m
%
% 1. For each (g,a), run the group-level iterative procedure
% 2. Save final group-level Ceff & error metrics
% 3. Find optimal (g,a) by min error or max correlation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxNumCompThreads(12);

if ~exist('GRIDON','var')
    GRIDON = false;
end

% --------------- Define ranges ----------------
if GRIDON
   Gvals = 0.01 : 0.01 : 2.0;   % optional grid over g (GRIDON = 1)
   Avals = -0.02;
else
   Gvals = 0.2;
   Avals = -0.02;
end

% --------------- Define parameters ------------
N        = str2double(ATLASNAME); %%%% 416
TR       = 2.000;  % Repetition time
sigma    = 0.01;   % noise amplitude
epsFC    = 0.0004;
epsFCtau = 0.0001;
maxC     = 0.2;
max_iters= 5000;

% --------------- Load data (SC, TS, etc.) -----

% Paths, atlas-specific filenames and the run tag all come from kasp_paths.m,
% shared with run_hopf_subject_refinement.m. Set KASP_DATA and KASP_RUN_TAG.
P = kasp_paths(ATLASNAME);
basepath     = P.basepath;
data_dir     = P.data_dir;
res_dir      = P.res_dir;
derived_data = P.derived_dir;      % this run's own folder, never data/derived_data
tspath       = P.tspath;

% Load SC
load(P.sc_file);
if size(SC, 2) ~= N
    error('SC in %s is %dx%d but ATLASNAME says %d parcels.', P.sc_file, size(SC,2), size(SC,3), N);
end

SC = squeeze(mean(SC,1));
SC     = SC / max(SC(:))*maxC;

ignoreFile = P.ignore_file;
load(ignoreFile, 'IgnoreNodes');
IgnoreNodes = double(IgnoreNodes(:))';
indexN = setdiff(1:N, IgnoreNodes);
fprintf('IgnoreNodes: %d dropped, %d retained\n', numel(IgnoreNodes), numel(indexN));

disp(IgnoreNodes)

indexN = setdiff(1:N, IgnoreNodes);
SC     = SC(indexN,indexN);

disp('Loaded structural connectivity')

% Load subjects name.
% load_subject_list applies the participant exclusions AND the medication split,
% keeping `subjects` and `is_med` aligned. Do not filter by is_med again here.
% The participant exclusions come from the SAME file as the node list
% (IgnoreNodes.mat, field ExcludeSubjects), so the two cannot diverge.
[subjects, subinfo] = load_subject_list(basepath, GROUP, MED, ignoreFile);

% Load timeseries
NSUB = length(subjects);
TS = [];  % NSUB x N x T

for iSub = 1:NSUB
    subID = subjects{iSub};
    fname = fullfile(tspath, [subID P.ts_suffix]);

    if isfile(fname)
        data = load(fname);
        % Get the timeseries variable (node x time)

        ts_sub = data.ts;  % node x time

        % ts_sub = data.ts;  % node x time
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

    % Preallocate: building this by assignment mis-shapes the array if seed 1
    % happens to be a node with no data.
    signal_filt = nan(Nvalid, size(ts,2));

    for seed=1:Nvalid
        if all(isnan(ts(seed,:)))
            continue                      % leave as NaN; filtfilt rejects non-finite
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
            % Without this guard max() on an all-NaN spectrum returns index 1,
            % assigning freq(1) = 0 -- a silently wrong intrinsic frequency.
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
nanmask = squeeze(any(isnan(TS), 3));                 % NSUB x Nvalid
fprintf('missing node-subject pairs: %d of %d (%.2f%%); min subjects per node: %d of %d\n', ...
        sum(nanmask(:)), numel(nanmask), 100*mean(nanmask(:)), ...
        min(sum(~nanmask,1)), NSUB);

f_diff = mean(f_diff_sub, 1, 'omitnan');
if any(~isfinite(f_diff))
    error('f_diff is non-finite at %d node(s) -- no subject has data there.', ...
          sum(~isfinite(f_diff)));
end
wo = 2*pi * f_diff(:);

save(fullfile(derived_data, sprintf('wo_group-%s_tau_%d_med-%d_atlas-%s.mat', GROUP, Tau, MED, ATLASNAME)), 'f_diff');

disp('Computed frequencies from subject timeseries')

[FCemp_group, COVtauemp_group, FCPB, COVtauPB] = compute_group_empirical(TS, Tau, indexN);

v = fullfile(derived_data, sprintf('FC_group-%s_tau-%d_med-%d.mat',GROUP,Tau,MED));
save (v,'FCPB','COVtauPB')

disp('NAN values for FCemp')
[r, c] = find(isnan(FCemp_group));
for k = 1:numel(r)
    disp(fprintf('NaN at row %d, column %d\n', r(k), c(k)));
end

disp('NAN values for COVtauemp_group')
[r, c] = find(isnan(COVtauemp_group));
for k = 1:numel(r)
    disp(fprintf('NaN at row %d, column %d\n', r(k), c(k)));
end

disp('Computed fc and covtau')

% lags for HopfModel_LNA in seconds
lags_model = [Tau*TR];

% --------------- Prepare storage --------------
nA = length(Avals);
nG = length(Gvals);
ErrorMat  = zeros(nG,nA);
CeffStore = cell(nG,nA);
FitStore  = cell(nG,nA);          % per-grid-point diagnostics

if GRIDON && nG > 1
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', 12);   % NumThreads defaults to 1 -> 12 cores
    end
end

% --------------- Loop over G, A ---------------
if GRIDON && nG > 1
	parfor iG = 1:nG
	  gVal = Gvals(iG);
	  ceffRow = cell(1,nA);
	  errRow  = zeros(1,nA);
	  fitRow  = cell(1,nA);
	  for iA = 1:nA
	    aVal = Avals(iA);
	    aVec = aVal * ones(Nvalid,1);

	    fprintf('g=%.6f a=%.6f\n', gVal, Avals(iA));

	    % === Run group-level model ===
	    [Ceff_group, errFC, errCOVtau, errTotal, FCsim, COVtauSim] = fit_group_level(...
	       SC, aVec, gVal, wo, sigma, ...
	       FCemp_group, COVtauemp_group, ...
	       epsFC, epsFCtau, max_iters, maxC, lags_model);

	    ceffRow{iA} = Ceff_group;
	    errRow(iA)  = errTotal;
	    fitRow{iA}  = struct('errFC',errFC, 'errCOVtau',errCOVtau, ...
                           'FCsim',FCsim, 'COVtauSim',COVtauSim);
	  end
	  CeffStore(iG,:) = ceffRow;     % iG alone in the first subscript, ':' in the second
	  ErrorMat(iG,:)  = errRow;
	  FitStore(iG,:)  = fitRow;
	end
else
	for iG = 1:nG
	  gVal = Gvals(iG);
	  for iA = 1:nA
	    aVal = Avals(iA);
	    aVec = aVal * ones(Nvalid,1);
            fprintf('g=%.6f a=%.6f\n', gVal, Avals(iA));

	    % === Run group-level model ===
	    [Ceff_group, errFC, errCOVtau, errTotal, FCsim, COVtauSim] = fit_group_level(...
	       SC, aVec, gVal, wo, sigma, ...
	       FCemp_group, COVtauemp_group, ...
	       epsFC, epsFCtau, max_iters, maxC, lags_model);

            CeffStore{iG,iA} = Ceff_group;
            ErrorMat(iG,iA)  = errTotal;
            FitStore{iG,iA}  = struct('errFC',errFC, 'errCOVtau',errCOVtau, ...
                                     'FCsim',FCsim, 'COVtauSim',COVtauSim);
	  end
   end
end

% --------------- Find best (g*, a*) -----------
[minVal, idx] = min(ErrorMat(:));  % lower = better
[iGbest, iAbest] = ind2sub(size(ErrorMat), idx);
gBest = Gvals(iGbest);
aBest = Avals(iAbest);
Ceff_best = CeffStore{iGbest, iAbest};

best      = FitStore{iGbest, iAbest};      % <-- add from here
errFC     = best.errFC;
errCOVtau = best.errCOVtau;
FCsim     = best.FCsim;
COVtauSim = best.COVtauSim;

disp('====================================================');
disp(['Best group-level parameters => g=' num2str(gBest) ...
      ', a=' num2str(aBest) ', error=' num2str(minVal)]);
disp('====================================================');

% --------------- Expand Ceff_best to full size (416x416) with NaN ------
Ceff_best_full = nan(N, N);  % N=416
Ceff_best_full(indexN, indexN) = Ceff_best;  % Fill in valid nodes

% --------------- Save all results -------------
results_dir = sprintf('%s/results_group_only/', res_dir);
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

results_file = fullfile(results_dir, ...
    sprintf('%s_group-%s_tau-%d_med-%d_atlas-%s.mat', ...
        ternary(GRIDON,'gridGroupResults','no_grid_GroupResults'), ...
        GROUP, Tau, MED, ATLASNAME));

code_commit = P.code_commit;      % provenance: the code that produced this file
code_dirty  = P.code_dirty;
run_tag     = P.run_tag;

% save final simulated FC and COVtau
save(results_file, ...
     'Gvals','Avals','ErrorMat','CeffStore','FitStore', ...
     'gBest','aBest','Ceff_best','Ceff_best_full', 'subjects', 'subinfo',... 
     'minVal','sigma','IgnoreNodes', 'indexN', 'errFC','errCOVtau', 'FCsim', 'COVtauSim', ...
     'code_commit', 'code_dirty', 'run_tag');

%--- Helper function ---%
function out = ternary(cond,trueVal,falseVal)
    if cond, out = trueVal; else out = falseVal; end
end
