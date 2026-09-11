% Generate fixed-parameter L0-smoothing supervision while preserving the
% existing train/test/test-200 split created in 20260813.

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

l0LibraryDir = 'D:\nuoqi\decoupled-learning-fan\DecoupleLearning-master\smoothing_method\L0smoothing_SIG_11';
if ~exist(l0LibraryDir, 'dir')
    error('L0 smoothing library not found: %s', l0LibraryDir);
end
addpath(l0LibraryDir);
if exist('L0Smoothing', 'file') ~= 2
    error('L0Smoothing.m is not available on the MATLAB path.');
end

sourceBasePath = 'D:\nuoqi\S2R-HDR\data\20260813';
baseOutputPath = 'D:\nuoqi\S2R-HDR\data\20260819';
lambda = 0.02;
kappa = 2.0;
parameterTag = 'l0p020';
numWorkers = 16;

splits = [ ...
    struct( ...
        'name', 'train', ...
        'sourceInputDir', fullfile(sourceBasePath, 'traininput-l1p000-a1p600'), ...
        'outputInputDir', fullfile(baseOutputPath, ['traininput-', parameterTag]), ...
        'outputLabelDir', fullfile(baseOutputPath, ['trainlabel-', parameterTag]) ...
    ), ...
    struct( ...
        'name', 'test', ...
        'sourceInputDir', fullfile(sourceBasePath, 'testinput-l1p000-a1p600'), ...
        'outputInputDir', fullfile(baseOutputPath, ['testinput-', parameterTag]), ...
        'outputLabelDir', fullfile(baseOutputPath, ['testlabel-', parameterTag]) ...
    ), ...
    struct( ...
        'name', 'test-200', ...
        'sourceInputDir', fullfile(sourceBasePath, 'testinput-l1p000-a1p600-200'), ...
        'outputInputDir', fullfile(baseOutputPath, ['testinput-', parameterTag, '-200']), ...
        'outputLabelDir', fullfile(baseOutputPath, ['testlabel-', parameterTag, '-200']) ...
    ) ...
];

if ~exist(baseOutputPath, 'dir')
    mkdir(baseOutputPath);
end

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local', numWorkers);
end

for splitIndex = 1:numel(splits)
    splitSpec = splits(splitIndex);
    if ~exist(splitSpec.sourceInputDir, 'dir')
        error('Source input directory not found: %s', splitSpec.sourceInputDir);
    end
    if ~exist(splitSpec.outputInputDir, 'dir')
        mkdir(splitSpec.outputInputDir);
    end
    if ~exist(splitSpec.outputLabelDir, 'dir')
        mkdir(splitSpec.outputLabelDir);
    end

    files = list_tiff_files(splitSpec.sourceInputDir);
    if isempty(files)
        error('No TIFF files found in %s', splitSpec.sourceInputDir);
    end

    parfor k = 1:numel(files)
        sourcePath = fullfile(splitSpec.sourceInputDir, files(k).name);
        logImg = double(imread(sourcePath));
        if ndims(logImg) == 3
            logImg = logImg(:, :, 1);
        end

        filteredImg = L0Smoothing(logImg, lambda, kappa);
        [~, baseName, ~] = fileparts(files(k).name);
        if ~endsWith(baseName, '-log')
            error('Expected a -log input filename, got: %s', files(k).name);
        end
        sampleName = baseName(1:end - numel('-log'));

        inputPath = fullfile(splitSpec.outputInputDir, [sampleName, '-log.tiff']);
        labelPath = fullfile(splitSpec.outputLabelDir, ...
            [sampleName, '-L0smooth-', parameterTag, '.tiff']);
        write_single_channel_tiff(inputPath, logImg);
        write_single_channel_tiff(labelPath, filteredImg);
    end

    outputInputs = list_tiff_files(splitSpec.outputInputDir);
    outputLabels = list_tiff_files(splitSpec.outputLabelDir);
    if numel(outputInputs) ~= numel(files) || numel(outputLabels) ~= numel(files)
        error('%s output count mismatch: expected %d, got input=%d label=%d', ...
            splitSpec.name, numel(files), numel(outputInputs), numel(outputLabels));
    end
    fprintf('%s complete: %d input/label pairs\n', splitSpec.name, numel(files));
end

fprintf('Done: L0 lambda=%.3f, kappa=%.1f, output=%s\n', lambda, kappa, baseOutputPath);

function files = list_tiff_files(directory)
    files = [dir(fullfile(directory, '*.tiff')); dir(fullfile(directory, '*.tif'))];
    [~, order] = sort({files.name});
    files = files(order);
end

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
