function run_generate_rtv_voc2012()
%RUN_GENERATE_RTV_VOC2012 Generate the fixed-parameter RTV VOC2012 dataset.
% Open this file in MATLAB and click Run. Paths are resolved relative to this
% file, so the current MATLAB working directory does not need to be changed.

thisFile = mfilename('fullpath');
projectDir = fileparts(thisFile);
decoupledLearningDir = fileparts(projectDir);
haloFreeRoot = fileparts(decoupledLearningDir);
researchCodeRoot = fileparts(haloFreeRoot);

inputDir = fullfile(haloFreeRoot, 'datasets', 'VOC2012', ...
    'VOC2012_trainval', 'JPEGImages');
outputDir = fullfile(haloFreeRoot, 'datasets', 'VOC2012', ...
    'RTV_fixed_lambda_0p01');
operatorRepo = fullfile(researchCodeRoot, 'decoupled-learning-fan', ...
    'DecoupleLearning-master');
rtvDir = fullfile(operatorRepo, 'smoothing_method', 'RTV_sig_12');

% Fixed RTV target used for this single-task model.
rtvLambda = 0.01;
rtvSigma = 3.0;
rtvSharpness = 0.02;
rtvMaxIter = 4;

expectedImageCount = 11540;
valRatio = 0.10;
testRatio = 0.10;
splitSeed = 20261008;
numWorkers = 24;

assert(isfolder(inputDir), 'VOC2012 input directory not found: %s', inputDir);
assert(isfile(fullfile(rtvDir, 'tsmooth.m')), ...
    'RTV implementation not found: %s', fullfile(rtvDir, 'tsmooth.m'));
assert(isfile(fullfile(projectDir, 'generate_operator_dataset_matlab.m')), ...
    'Dataset generator is missing from: %s', projectDir);

imageFiles = dir(fullfile(inputDir, '*.jpg'));
if numel(imageFiles) ~= expectedImageCount
    error('Expected %d VOC2012 trainval JPEGs, found %d in %s.', ...
        expectedImageCount, numel(imageFiles), inputDir);
end

if isfolder(outputDir)
    existing = dir(fullfile(outputDir, '*'));
    existing = existing(~ismember({existing.name}, {'.', '..'}));
    if ~isempty(existing)
        error(['Output directory already contains files: %s\n' ...
            'Move or rename that directory before starting a fresh run.'], outputDir);
    end
end

addpath(projectDir, '-begin');
addpath(rtvDir, '-begin');
clear tsmooth;
rehash;
resolvedRtv = which('tsmooth');
if isempty(resolvedRtv) || isempty(strfind(resolvedRtv, fullfile('RTV_sig_12', 'tsmooth.m'))) %#ok<STREMP>
    error('MATLAB did not resolve tsmooth.m from the expected RTV folder. Resolved: %s', resolvedRtv);
end

fprintf('VOC2012 source images: %d\n', numel(imageFiles));
fprintf('RTV parameters: lambda=%.5f, sigma=%.2f, sharpness=%.3f, maxIter=%d\n', ...
    rtvLambda, rtvSigma, rtvSharpness, rtvMaxIter);
fprintf('Input:  %s\n', inputDir);
fprintf('Output: %s\n', outputDir);
fprintf('Split:  %.0f%% train / %.0f%% val / %.0f%% test (seed %d)\n', ...
    100 * (1 - valRatio - testRatio), 100 * valRatio, 100 * testRatio, splitSeed);
fprintf('Parallel workers: %d\n', numWorkers);

generate_operator_dataset_matlab( ...
    'inputDir', inputDir, ...
    'outputDir', outputDir, ...
    'operator', 'rtv', ...
    'mode', 'y', ...
    'variantsPerImage', 1, ...
    'valRatio', valRatio, ...
    'testRatio', testRatio, ...
    'seed', splitSeed, ...
    'numWorkers', numWorkers, ...
    'outputFormat', 'tiff', ...
    'overwrite', false, ...
    'rtvLambda', rtvLambda, ...
    'rtvSigma', rtvSigma, ...
    'rtvSharpness', rtvSharpness, ...
    'rtvMaxIter', rtvMaxIter);

fprintf('\nRTV dataset generation finished.\n');
fprintf('Dataset: %s\n', outputDir);
end
