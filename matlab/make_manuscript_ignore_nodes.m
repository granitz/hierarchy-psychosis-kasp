% MAKE_MANUSCRIPT_IGNORE_NODES  IgnoreNodes.mat for the paper's main analysis.
%
% The pipeline reads both the parcels to drop and the participants to exclude
% from IgnoreNodes.mat. For the main analysis (data/tseries_2) these are the 45
% parcels with coverage below 50% and no further participant excluded; this
% writes that file.
%
% Usage (KASP_TIMESERIES as for the run):
%   setenv('KASP_TIMESERIES', '<KASP_DATA>/data/tseries_2/mat_files')
%   run('make_manuscript_ignore_nodes.m')
%
% Refuses to overwrite an existing IgnoreNodes.mat that differs from this one.

tsdir = getenv('KASP_TIMESERIES');
if isempty(tsdir)
    error('Set KASP_TIMESERIES to the timeseries directory the run will use.');
end
out = fullfile(tsdir, 'IgnoreNodes.mat');

% The 45 parcels excluded for low coverage (MATLAB,
% one-based parcel indices into the 416-parcel atlas, subcortex first).
IgnoreNodes = [28,31, 85, 130,131,132,133,134, 135,136,137,138,139,140,141,142,150, ...
               165,166,167,181, 217,226,229,237,287, 335,336,337,338,339,340,341,342, ...
               343,344,345,346,347,354,355,357,358,383,384];
ExcludeSubjects     = cell(1, 0);     % no further exclusions at this stage
min_region_coverage = NaN;            % not derived from coverage: hardcoded list
drop_fraction       = NaN;
source = 'main analysis: 45 parcels with coverage < 50%; no participants excluded';

assert(numel(IgnoreNodes) == 45 && numel(unique(IgnoreNodes)) == 45);

if isfile(out)
    old = load(out);
    same = isfield(old, 'IgnoreNodes') && isequal(sort(old.IgnoreNodes(:)'), IgnoreNodes) ...
        && isfield(old, 'ExcludeSubjects') && isempty(old.ExcludeSubjects);
    if same
        fprintf('%s already holds the manuscript list; nothing to do.\n', out);
        return
    end
    error(['%s exists and differs from the manuscript list. It was not written ' ...
           'by this script; move it aside first rather than overwrite it.'], out);
end

save(out, 'IgnoreNodes', 'ExcludeSubjects', 'min_region_coverage', ...
     'drop_fraction', 'source');
fprintf('wrote %s: %d parcels dropped, %d participants excluded\n', ...
        out, numel(IgnoreNodes), numel(ExcludeSubjects));
