function generate_operator_dataset_matlab(varargin)
%GENERATE_OPERATOR_DATASET_MATLAB Generate supervised operator-imitation data.
%
% This is the MATLAB version of the ordinary-image CELiNet data flow:
%   ordinary image -> im2double -> classical operator -> paired label.
% It intentionally does not perform the HDR log2 conversion or per-image
% log normalization used by the HDR WLS scripts. The WLS implementation
% itself still computes log(I + eps) internally, as defined by wlsFilter.m.
%
% Example:
% generate_operator_dataset_matlab( ...
%   'inputDir', '/path/to/images', ...
%   'outputDir', '/path/to/operator_dataset', ...
%   'operator', 'wls', 'mode', 'y', 'variantsPerImage', 7, ...
%   'operatorRoot', '/path/to/DecoupleLearning-master');
%
% The input and label files are saved under train/val/test. A JSONL file
% records source image, sampled parameters and paired relative paths.

p = inputParser;
addParameter(p, 'inputDir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'outputDir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'operator', 'wls', @(x) ischar(x) || isstring(x));
addParameter(p, 'mode', 'y', @(x) ischar(x) || isstring(x));
addParameter(p, 'variantsPerImage', [], @(x) isempty(x) || (isscalar(x) && x >= 1));
addParameter(p, 'valRatio', 0.1, @(x) isscalar(x) && x >= 0);
addParameter(p, 'testRatio', 0.1, @(x) isscalar(x) && x >= 0);
addParameter(p, 'seed', 20261008, @(x) isscalar(x));
addParameter(p, 'outputFormat', 'tiff', @(x) ischar(x) || isstring(x));
addParameter(p, 'overwrite', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'operatorRoot', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'lambdaBase', 1.0, @isscalar);
addParameter(p, 'lambdaLogRange', [-log(10), log(10)], @(x) numel(x) == 2);
addParameter(p, 'alphaRange', [1.0, 1.6], @(x) numel(x) == 2);
addParameter(p, 'gaussianSigmaRange', [0.8, 4.0], @(x) numel(x) == 2);
addParameter(p, 'rgfSigmaRange', [1.0, 8.0], @(x) numel(x) == 2);
addParameter(p, 'l0LambdaBase', 0.02, @isscalar);
addParameter(p, 'l0LogRange', [-log(10), log(10)], @(x) numel(x) == 2);
addParameter(p, 'rtvLambda', 0.01, @(x) isscalar(x) && x > 0);
addParameter(p, 'rtvSigma', 3.0, @(x) isscalar(x) && x > 0);
addParameter(p, 'rtvSharpness', 0.02, @(x) isscalar(x) && x > 0);
addParameter(p, 'rtvMaxIter', 4, @(x) isscalar(x) && x >= 1 && mod(x, 1) == 0);
addParameter(p, 'numWorkers', 24, @(x) isscalar(x) && x >= 1 && mod(x, 1) == 0);
parse(p, varargin{:});
opts = p.Results;

opts.inputDir = char(opts.inputDir);
opts.outputDir = char(opts.outputDir);
opts.operator = lower(char(opts.operator));
opts.mode = lower(char(opts.mode));
opts.outputFormat = lower(char(opts.outputFormat));
if isempty(opts.inputDir) || isempty(opts.outputDir)
    error('inputDir and outputDir are required.');
end
if opts.valRatio + opts.testRatio >= 1
    error('valRatio + testRatio must be smaller than 1.');
end
if ~ismember(opts.operator, {'wls','l0','rtv','rolling_guidance','gaussian','bilateral'})
    error('Unsupported operator: %s', opts.operator);
end
if ~ismember(opts.mode, {'y','gray','rgb'})
    error('mode must be y, gray or rgb.');
end
if ~ismember(opts.outputFormat, {'tiff','png'})
    error('outputFormat must be tiff or png.');
end

% Add the original CELiNet operator implementations when supplied.
if ~isempty(opts.operatorRoot) && isfolder(opts.operatorRoot)
    addpath(genpath(opts.operatorRoot));
end

files = list_image_files(opts.inputDir);
if isempty(files)
    error('No supported images found under %s.', opts.inputDir);
end
if isfolder(opts.outputDir) && ~opts.overwrite
    existing = dir(fullfile(opts.outputDir, '*'));
    existing = existing(~ismember({existing.name}, {'.', '..'}));
    if ~isempty(existing)
        error('Output directory is not empty. Pass ''overwrite'', true to reuse it.');
    end
end
if ~isfolder(opts.outputDir), mkdir(opts.outputDir); end

if isempty(opts.variantsPerImage)
    % A single deterministic target is the default for the single-task
    % texture-removal setup. Use variantsPerImage explicitly for sweeps.
    variants = 1;
else
    variants = opts.variantsPerImage;
end

rng(opts.seed, 'twister');
order = randperm(numel(files));
testCount = round(numel(files) * opts.testRatio);
valCount = round(numel(files) * opts.valRatio);
testIdx = order(1:testCount);
valIdx = order(testCount + (1:valCount));
trainIdx = order(testCount + valCount + 1:end);
splitNames = {'test','val','train'};

processingIndices = [testIdx(:); valIdx(:); trainIdx(:)];
processingSplits = [repmat({'test'}, numel(testIdx), 1); ...
    repmat({'val'}, numel(valIdx), 1); repmat({'train'}, numel(trainIdx), 1)];
processingOrder = kron((1:numel(processingIndices))', ones(variants, 1));
jobImageIndices = processingIndices(processingOrder);
jobSplitNames = processingSplits(processingOrder);
jobVariants = repmat(0:variants-1, 1, numel(processingIndices))';
jobFiles = files(jobImageIndices);
jobFileFolders = {jobFiles.folder}';
jobFileNames = {jobFiles.name}';
numJobs = numel(jobImageIndices);
metadataLines = cell(numJobs, 1);
for s = 1:numel(splitNames)
    inputDir = fullfile(opts.outputDir, splitNames{s}, 'input');
    labelDir = fullfile(opts.outputDir, splitNames{s}, 'label');
    if ~isfolder(inputDir), mkdir(inputDir); end
    if ~isfolder(labelDir), mkdir(labelDir); end
end

% Each worker writes a unique pair; metadata is written by the client later.
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', opts.numWorkers);
elseif pool.NumWorkers ~= opts.numWorkers
    fprintf('Restarting existing pool (%d workers) with %d workers.\n', ...
        pool.NumWorkers, opts.numWorkers);
    delete(pool);
    pool = parpool('local', opts.numWorkers);
end
fprintf('Using MATLAB parallel pool with %d workers.\n', pool.NumWorkers);
generatorPath = mfilename('fullpath');
rtvPath = which('tsmooth');
if isempty(rtvPath)
    error('Cannot find tsmooth.m. Add the original RTV folder to the MATLAB path.');
end
addAttachedFiles(pool, {generatorPath, rtvPath});
queue = parallel.pool.DataQueue;
progress_tick(-1, numJobs);
afterEach(queue, @(message) progress_tick(message, numJobs));
parfor job = 1:numJobs
    split = jobSplitNames{job};
    variant = jobVariants(job);
    sourcePath = fullfile(jobFileFolders{job}, jobFileNames{job});
    image = load_image_mode(sourcePath, opts.mode);
    sourceRel = relative_path(sourcePath, opts.inputDir);
    sourceId = regexprep(strrep(sourceRel, filesep, '__'), '\.[^\.]+$', '');
    rng(opts.seed + job, 'twister');
    [label, params] = apply_operator(image, opts);
    stem = sprintf('%s__%s__v%02d', sourceId, opts.operator, variant);
    inputFile = save_array(fullfile(opts.outputDir, split, 'input', stem), image, opts.outputFormat);
    labelFile = save_array(fullfile(opts.outputDir, split, 'label', stem), label, opts.outputFormat);
    % Build the JSON in a helper function. A struct whose fields are added
    % one by one inside parfor cannot be classified reliably by MATLAB.
    metadataLines{job} = make_metadata_line( ...
        split, sourceRel, variant, opts.operator, params, opts.mode, ...
        relative_path(inputFile, opts.outputDir), ...
        relative_path(labelFile, opts.outputDir));
    send(queue, 1);
end

metadataPath = fullfile(opts.outputDir, 'metadata.jsonl');
metadata = fopen(metadataPath, 'w');
if metadata < 0, error('Cannot open metadata file for writing: %s', metadataPath); end
cleanupObj = onCleanup(@() fclose(metadata));
for job = 1:numJobs
    fprintf(metadata, '%s\n', metadataLines{job});
end
pairCount = numJobs;
fprintf('Generated %d pairs from %d source images.\n', pairCount, numel(files));
fprintf('Output: %s\n', opts.outputDir);
end

function files = list_image_files(root)
% Recursive image listing without requiring newer dir(..., '**') syntax.
files = struct('folder', {}, 'name', {});
entries = dir(root);
for k = 1:numel(entries)
    name = entries(k).name;
    if entries(k).isdir
        if ~strcmp(name, '.') && ~strcmp(name, '..')
            child = list_image_files(fullfile(root, name));
            files = [files, child]; %#ok<AGROW>
        end
    elseif is_image_name(name)
        item.folder = root;
        item.name = name;
        files(end + 1) = item; %#ok<AGROW>
    end
end
end

function yes = is_image_name(name)
[~,~,ext] = fileparts(name);
yes = ismember(lower(ext), {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'});
end

function image = load_image_mode(path, mode)
image = im2double(imread(path));
if ndims(image) == 2
    if strcmp(mode, 'rgb'), image = repmat(image, [1 1 3]); end
    return;
end
if size(image, 3) > 3, image = image(:,:,1:3); end
switch mode
    case 'rgb'
        % Keep RGB channels.
    case 'gray'
        image = rgb2gray(image);
    case 'y'
        ycbcr = rgb2ycbcr(image);
        image = ycbcr(:,:,1);
end
image = min(max(image, 0), 1);
end

function [label, params] = apply_operator(image, opts)
params = struct();
switch opts.operator
    case 'wls'
        lambda = opts.lambdaBase * exp(rand_range(opts.lambdaLogRange));
        alpha = rand_range(opts.alphaRange);
        params.lambda = lambda; params.alpha = alpha;
        if ndims(image) == 3
            label = zeros(size(image));
            for c = 1:size(image,3), label(:,:,c) = wlsFilter(image(:,:,c), lambda, alpha); end
        else
            label = wlsFilter(image, lambda, alpha);
        end
    case 'l0'
        lambda = opts.l0LambdaBase * exp(rand_range(opts.l0LogRange));
        params.lambda = lambda;
        label = L0Smoothing(im2uint8(image), lambda);
        label = im2double(label);
    case 'rtv'
        lambda = opts.rtvLambda;
        sigma = opts.rtvSigma;
        sharpness = opts.rtvSharpness;
        maxIter = opts.rtvMaxIter;
        params.lambda = lambda;
        params.sigma = sigma;
        params.sharpness = sharpness;
        params.maxIter = maxIter;
        label = tsmooth(im2uint8(image), lambda, sigma, sharpness, maxIter);
        label = im2double(label);
    case 'rolling_guidance'
        sigma = rand_range(opts.rgfSigmaRange);
        params.sigma = sigma; params.sigma_r = 0.05; params.iterations = 4;
        label = RollingGuidanceFilter(image, sigma, 0.05, 4);
    case 'gaussian'
        sigma = rand_range(opts.gaussianSigmaRange);
        params.sigma = sigma;
        if exist('imgaussfilt', 'file')
            label = imgaussfilt(image, sigma);
        else
            radius = max(1, ceil(3*sigma));
            x = -radius:radius; kernel = exp(-(x.^2)/(2*sigma^2)); kernel = kernel/sum(kernel);
            label = imfilter(imfilter(image, kernel, 'replicate'), kernel', 'replicate');
        end
    case 'bilateral'
        params.sigma = 3;
        if exist('imbilatfilt', 'file')
            label = imbilatfilt(image, 0.1, 3);
        else
            error('bilateral requires imbilatfilt (Image Processing Toolbox).');
        end
end
label = min(max(label, 0), 1);
end

function line = make_metadata_line(split, source, variant, operator, params, mode, inputFile, labelFile)
%MAKE_METADATA_LINE Construct one complete metadata record outside parfor.
record = struct( ...
    'split', split, ...
    'source', source, ...
    'variant', variant, ...
    'operator', operator, ...
    'parameters', params, ...
    'mode', mode, ...
    'input', inputFile, ...
    'label', labelFile);
line = jsonencode(record);
end

function value = rand_range(range)
value = range(1) + rand() * (range(2) - range(1));
end

function progress_tick(message, totalJobs)
persistent completedJobs
if message < 0
    completedJobs = 0;
    fprintf('RTV generation progress: 0/%d jobs\n', totalJobs);
    return;
end
completedJobs = completedJobs + 1;
if mod(completedJobs, 100) == 0 || completedJobs == totalJobs
    fprintf('RTV generation progress: %d/%d jobs\n', completedJobs, totalJobs);
end
end

function outputPath = save_array(stem, image, format)
outDir = fileparts(stem);
if ~isfolder(outDir), mkdir(outDir); end
outputPath = [stem, '.', format];
if strcmp(format, 'png')
    imwrite(uint8(round(min(max(image,0),1) * 255)), outputPath);
else
    write_float_tiff(outputPath, image);
end
end

function write_float_tiff(path, image)
image = single(image);
if ndims(image) == 2
    t = Tiff(path, 'w');
    tags.ImageLength = size(image,1); tags.ImageWidth = size(image,2);
    tags.SampleFormat = Tiff.SampleFormat.IEEEFP; tags.Photometric = Tiff.Photometric.MinIsBlack;
    tags.BitsPerSample = 32; tags.SamplesPerPixel = 1;
    tags.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; tags.Compression = Tiff.Compression.None;
    t.setTag(tags); t.write(image); t.close();
else
    if size(image, 3) ~= 3
        error('Expected an HxWx3 RGB image, got size %s.', mat2str(size(image)));
    end
    t = Tiff(path, 'w');
    tags.ImageLength = size(image,1); tags.ImageWidth = size(image,2);
    tags.SampleFormat = Tiff.SampleFormat.IEEEFP; tags.Photometric = Tiff.Photometric.RGB;
    tags.BitsPerSample = 32; tags.SamplesPerPixel = 3;
    tags.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; tags.Compression = Tiff.Compression.None;
    t.setTag(tags); t.write(image); t.close();
end
end

function rel = relative_path(path, root)
path = char(path); root = char(root);
prefix = [root, filesep];
if strncmp(path, prefix, numel(prefix)), rel = path(numel(prefix)+1:end); else, rel = path; end
rel = strrep(rel, filesep, '/');
end
