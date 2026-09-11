clear; clc;

% Generate test_vivo log input and WLS label directly from linear HDR TIFFs.
% The output naming follows the existing test_vivo pipeline convention.

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

inputDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
outputRoot = 'D:\nuoqi\S2R-HDR\data\test_vivo';
lambda = 0.5;
alpha = 1.4;
parameterTag = 'l0p500-a1p400';
numWorkers = 16;

outputInputDir = fullfile(outputRoot, ['testinput-', parameterTag]);
outputLabelDir = fullfile(outputRoot, ['testlabel-', parameterTag]);
ensure_dir(outputRoot);
ensure_dir(outputInputDir);
ensure_dir(outputLabelDir);

files = [dir(fullfile(inputDir, '*.tiff')); dir(fullfile(inputDir, '*.tif'))];
if isempty(files)
    error('No HDR files found in %s', inputDir);
end

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local', numWorkers);
end

parfor k = 1:numel(files)
    sourcePath = fullfile(files(k).folder, files(k).name);
    rgb = double(imread(sourcePath));
    if ndims(rgb) ~= 3 || size(rgb, 3) ~= 3
        error('Expected a 3-channel HDR image: %s', sourcePath);
    end

    maxVal = max(rgb(:));
    if maxVal > 0
        rgb = rgb / maxVal;
    end
    intensity = max(0.299 * rgb(:, :, 1) + ...
                    0.587 * rgb(:, :, 2) + ...
                    0.114 * rgb(:, :, 3), 1e-10);

    logImg = 1024 * log(intensity);
    logMin = min(logImg(:));
    logMax = max(logImg(:));
    logRange = logMax - logMin;
    if logRange < 1e-12
        logNorm = zeros(size(logImg));
    else
        logNorm = (logImg - logMin) / logRange;
    end
    filteredImg = wls(logNorm, lambda, alpha);

    [~, sampleName, ~] = fileparts(files(k).name);
    inputPath = fullfile(outputInputDir, [sampleName, '-log.tiff']);
    labelPath = fullfile(outputLabelDir, [sampleName, '-WLS-', parameterTag, '.tiff']);
    write_single_channel_tiff(inputPath, logNorm);
    write_single_channel_tiff(labelPath, filteredImg);
end

outputInputs = dir(fullfile(outputInputDir, '*.tiff'));
outputLabels = dir(fullfile(outputLabelDir, '*.tiff'));
if numel(outputInputs) ~= numel(files) || numel(outputLabels) ~= numel(files)
    error('Output count mismatch: expected %d, got input=%d label=%d', ...
        numel(files), numel(outputInputs), numel(outputLabels));
end
fprintf('Done: %d pairs, lambda=%.3f, alpha=%.3f\n', numel(files), lambda, alpha);
fprintf('Input output: %s\nLabel output: %s\n', outputInputDir, outputLabelDir);

function write_single_channel_tiff(fileName, img)
    tiffFile = Tiff(fileName, 'w');
    tagstruct = struct();
    tagstruct.ImageLength = size(img, 1);
    tagstruct.ImageWidth = size(img, 2);
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;
    tiffFile.setTag(tagstruct);
    tiffFile.write(single(img));
    tiffFile.close();
end

function ensure_dir(pathstr)
    if ~exist(pathstr, 'dir')
        mkdir(pathstr);
    end
end
