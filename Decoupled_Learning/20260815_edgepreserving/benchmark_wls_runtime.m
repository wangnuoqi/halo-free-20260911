function benchmark_wls_runtime()
% Time the original MATLAB WLS solver on the edge-preserving test input.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, '..', 'total-pipeline-v1'));
dataDir = 'D:\ResearchCode\S2R-HDR\data\20260818\testinput-l0p500-a1p400-200';
inputPath = fullfile(dataDir, '9C4A0004-db1d4a14f3_cropped-log.tiff');
outputPath = fullfile(root, 'baseline', 'evaluation', 'wls_runtime_20261104.json');
if ~isfile(inputPath)
    error('WLS benchmark input not found: %s', inputPath);
end
image512 = double(imread(inputPath));
if ndims(image512) ~= 2 || ~isequal(size(image512), [512, 512])
    error('Expected the 512x512 single-channel WLS input: %s', inputPath);
end
lambda = 0.5;
alpha = 1.4;
fprintf('WLS 512x512 warmup...\n');
warm = wls(image512, lambda, alpha);
clear warm;
times512 = zeros(1, 3);
for k = 1:numel(times512)
    timer = tic;
    filtered = wls(image512, lambda, alpha);
    times512(k) = toc(timer);
    clear filtered;
    fprintf('WLS 512x512 repeat %d/%d: %.3f s\n', k, numel(times512), times512(k));
end
report = struct();
report.protocol = struct('implementation', 'original MATLAB total-pipeline-v1/wls.m', ...
    'lambda', lambda, 'alpha', alpha, 'timing', 'CPU sparse direct solve; input read excluded', ...
    'input_path', inputPath, 'timestamp', datestr(now, 'yyyy-mm-ddTHH:MM:SS'), ...
    'matlab_version', version, 'computer', computer, 'warmup_512', 1, 'repeats_512', numel(times512));
report.results = struct();
report.results.native_512 = struct('shape', [512, 512], 'status', 'ok', ...
    'seconds_samples', times512, 'seconds_median', median(times512));
save_report(outputPath, report);
fprintf('Preparing 3840x2160 resized input...\n');
image4k = imresize(image512, [2160, 3840], 'lanczos3');
fprintf('WLS 3840x2160 full-frame solve started.\n');
try
    timer = tic;
    filtered = wls(image4k, lambda, alpha);
    seconds4k = toc(timer);
    clear filtered;
    report.results.resized_4k = struct('shape', [2160, 3840], 'status', 'ok', ...
        'seconds_samples', seconds4k, 'seconds_median', seconds4k, ...
        'input_note', '512x512 test input resized to 3840x2160; synthetic 4K stress test');
catch exception
    report.results.resized_4k = struct('shape', [2160, 3840], 'status', 'error', ...
        'error', exception.message, ...
        'input_note', '512x512 test input resized to 3840x2160; synthetic 4K stress test');
end
save_report(outputPath, report);
fprintf('Saved WLS runtime report: %s\n', outputPath);
end

function save_report(path, report)
if ~exist(fileparts(path), 'dir')
    mkdir(fileparts(path));
end
fid = fopen(path, 'w');
if fid < 0
    error('Cannot write benchmark report: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(report, 'PrettyPrint', true));
end
