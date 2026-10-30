function [subjects, info] = load_subject_list(basepath, GROUP, MED, ignoreFile)
% LOAD_SUBJECT_LIST  Subject list for one group, with exclusions applied.
%
%   [subjects, info] = load_subject_list(basepath, GROUP, MED, ignoreFile)
%
%   GROUP       'ctrl' or 'sz'
%   MED         1 = medicated, 0 = unmedicated (ignored when GROUP is 'ctrl')
%   ignoreFile  path to the IgnoreNodes.mat written by make_ignore_nodes.py.
%               Its ExcludeSubjects field is the list of participants to
%               drop, so the participants dropped by the Python coverage step
%               and by the MATLAB fit cannot diverge.
%
% Exclusions are applied to `subjects` and `is_med` TOGETHER and BEFORE the
% medication split. is_med is positionally aligned to the full subject list, so
% filtering one without the other silently shifts every medication label after
% the first excluded patient.
%
% Called by run_hopf_grid_group.m and run_hopf_subject_refinement.m so both use
% an identical sample.

% ---- exclusions: read from IgnoreNodes.mat, never from this file ----------
% The participants to exclude are listed in the ExcludeSubjects field of the
% IgnoreNodes.mat that make_ignore_nodes.py writes next to the timeseries, so
% that the parcel list and the sample are decided together and travel together.
% They are deliberately not written here: this file is under version control
% and must not carry participant identifiers. The reasons for each exclusion
% belong in the decision log that sits with the data (kasp-decision-log.md).
%
% An ExcludeSubjects field that is present but empty means "exclude nobody".
% A missing field is an error rather than a silent "exclude nobody", because an
% older IgnoreNodes.mat without the field would otherwise quietly bring back
% participants a newer one excludes.
if nargin < 4 || isempty(ignoreFile)
    error('load_subject_list:noIgnoreFile', ...
          'Pass the IgnoreNodes.mat path: the exclusions are read from it.');
end
if ~isfile(ignoreFile)
    error('load_subject_list:noIgnoreFile', ...
          'ignoreFile does not exist: %s', ignoreFile);
end
V = load(ignoreFile);
if ~isfield(V, 'ExcludeSubjects')
    error('load_subject_list:noExcludeField', ...
          ['%s has no ExcludeSubjects field. Regenerate it with ' ...
           'make_ignore_nodes.py, listing the participants to exclude ' ...
           '(or none), so the node list and the sample travel together.'], ...
          ignoreFile);
end
if isempty(V.ExcludeSubjects)
    EXCLUDE = {};
else
    EXCLUDE = cellstr(string(V.ExcludeSubjects(:)'));
end
exclusion_source = ignoreFile;

f = fullfile(basepath, sprintf('subjects_%s_clean_good_coverage.mat', GROUP));
S = load(f, 'subjects');
subjects = S.subjects(:);
n_start  = numel(subjects);

if strcmpi(GROUP, 'sz')
    M = load(fullfile(basepath, 'is_med.mat'), 'is_med');
    is_med = M.is_med(:);

    if numel(is_med) ~= numel(subjects)
        error('load_subject_list:misaligned', ...
              'is_med has %d entries but the subject list has %d.', ...
              numel(is_med), numel(subjects));
    end

    keep     = ~ismember(subjects, EXCLUDE);
    subjects = subjects(keep);
    is_med   = is_med(keep);            % must be filtered with `subjects`

    subjects = subjects(is_med == MED);
else
    subjects = subjects(~ismember(subjects, EXCLUDE));
end

dropped = intersect(S.subjects(:), EXCLUDE);
info = struct('group', GROUP, 'med', MED, ...
              'n_before_exclusion', n_start, 'n', numel(subjects), ...
              'excluded', {dropped(:)'}, 'exclusion_list', {EXCLUDE}, ...
              'exclusion_source', exclusion_source);

% Counts only: this line goes to the run log, which must not carry identifiers.
% The identifiers themselves are in info.excluded, saved with the results.
fprintf('%s (med=%d): %d subjects (%d excluded by %s)\n', GROUP, MED, ...
        numel(subjects), numel(dropped), 'IgnoreNodes.mat ExcludeSubjects');
end
