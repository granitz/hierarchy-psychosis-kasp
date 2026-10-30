function P = kasp_paths(ATLASNAME)
% KASP_PATHS  Where the pipeline reads and writes, for one atlas and one run.
%
%   P = kasp_paths(ATLASNAME)      e.g. kasp_paths('416') or kasp_paths('116')
%
% Both run_hopf_grid_group.m and run_hopf_subject_refinement.m call this, so
% there is one definition of every path rather than one per atlas copy.
% Everything machine-specific comes from the environment, so no path is ever
% committed:
%
%   KASP_DATA          project root: subjects_*_clean_good_coverage.mat,
%                      is_med.mat and data/ (with the SC matrix)   [required]
%   KASP_RUN_TAG       name of this run                           [required]
%   KASP_TIMESERIES    parcellated timeseries + IgnoreNodes.mat
%                      (default <KASP_DATA>/data/tseries/mat_files)
%   KASP_RESULTS_ROOT  parent of all runs' results
%                      (default <KASP_DATA>/results)
%   KASP_LNA_DIR       directory holding HopfModel_LNA.m, if not on the path
%
% Every output of a run -- group fits, subject fits, and the derived FC and
% intrinsic frequencies -- goes under <KASP_RESULTS_ROOT>/<KASP_RUN_TAG>/.
% Nothing is written into data/, so two runs cannot overwrite each other's
% files.
%
% P also carries the git commit of this code and whether the working tree
% was modified, which the scripts save into every results file.

basepath = getenv('KASP_DATA');
if isempty(basepath)
    error('kasp_paths:noData', ...
          'Set KASP_DATA to the project root (see matlab/README.md).');
end
run_tag = getenv('KASP_RUN_TAG');
if isempty(run_tag)
    error('kasp_paths:noTag', ...
          ['Set KASP_RUN_TAG to name this run (e.g. main). ' ...
           'Results go to results/<tag>/, so runs cannot overwrite each other.']);
end

P.atlas    = char(ATLASNAME);
P.run_tag  = run_tag;
P.basepath = basepath;
P.data_dir = fullfile(basepath, 'data');

P.tspath = getenv('KASP_TIMESERIES');
if isempty(P.tspath)
    P.tspath = fullfile(P.data_dir, 'tseries', 'mat_files');
end
P.ignore_file = fullfile(P.tspath, 'IgnoreNodes.mat');

P.sc_file = fullfile(P.data_dir, sprintf( ...
    'atlas_SchaeferTian%s_sift_invnodevol_radius2_count_connectivity.mat', P.atlas));
P.ts_suffix = sprintf( ...
    '_task-rest_run-1_feature-corrMatrixICA24Motion_atlas-SchaeferTian%s_timeseries.mat', P.atlas);

root = getenv('KASP_RESULTS_ROOT');
if isempty(root)
    root = fullfile(basepath, 'results');
end
P.res_dir      = fullfile(root, run_tag);
P.group_dir    = fullfile(P.res_dir, 'results_group_only');
P.subject_dir  = fullfile(P.res_dir, 'results_subject_refinement');
P.derived_dir  = fullfile(P.res_dir, 'derived_data');

for d = {P.group_dir, P.subject_dir, P.derived_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

for f = {P.sc_file, P.ignore_file}
    if ~isfile(f{1})
        error('kasp_paths:missing', 'Required input not found: %s', f{1});
    end
end

% --- code on the path --------------------------------------------------------
here = fileparts(mfilename('fullpath'));
addpath(here);
lna = getenv('KASP_LNA_DIR');
if ~isempty(lna), addpath(lna); end
if exist('HopfModel_LNA', 'file') ~= 2
    error('kasp_paths:noLNA', ...
          ['HopfModel_LNA.m (Ponce-Alvarez & Deco) is not on the path. It is not ' ...
           'distributed here; set KASP_LNA_DIR to the folder holding it.']);
end

% --- provenance: which code produced this run ---------------------------------
[status, commit] = system(sprintf('git -C "%s" rev-parse HEAD', here));
if status == 0
    P.code_commit = strtrim(commit);
    [~, dirty] = system(sprintf('git -C "%s" status --porcelain --untracked-files=no', here));
    P.code_dirty = ~isempty(strtrim(dirty));
else
    P.code_commit = 'unknown (not a git checkout)';
    P.code_dirty = true;
end

fprintf('run %s | atlas %s | code %s%s\n', P.run_tag, P.atlas, ...
        P.code_commit(1:min(12, end)), repmat(' (modified)', 1, P.code_dirty));
fprintf('results -> %s\n', P.res_dir);
end
