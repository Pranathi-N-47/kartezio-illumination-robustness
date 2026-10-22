% Stable command-line entry point for the MATLAB baseline reference.
mode = getenv('KARTEZIO_BASELINE_MODE');
benchmark_root = getenv('KARTEZIO_BENCHMARK_ROOT');
output_dir = getenv('KARTEZIO_BASELINE_OUTPUT');
evaluation_dir = fileparts(mfilename('fullpath'));
addpath(evaluation_dir, '-begin');
clear baselines;
if isempty(mode), mode = 'full'; end
if isempty(benchmark_root), benchmark_root = fullfile(pwd, 'hardware_items'); end
if isempty(output_dir), output_dir = ''; end
baselines(mode, benchmark_root, output_dir);
