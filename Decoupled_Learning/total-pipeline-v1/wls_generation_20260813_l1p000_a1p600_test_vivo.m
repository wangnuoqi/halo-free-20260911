clear; clc;

% Generate test_vivo 0813 log input and WLS label from linear HDR TIFFs.
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

inputDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
outputRoot = 'D:\nuoqi\S2R-HDR\data\test_vivo';
lambda = 1.0;
alpha = 1.6;
parameterTag = 'l1p000-a1p600';
numWorkers = 16;

outputInputDir = fullfile(outputRoot, ['testinput-', parameterTag]);
outputLabelDir = fullfile(outputRoot, ['testlabel-', parameterTag]);
ensure_dir(outputInputDir); ensure_dir(outputLabelDir);
files = [dir(fullfile(inputDir, '*.tiff')); dir(fullfile(inputDir, '*.tif'))];
if isempty(files), error('No TIFF files found in %s', inputDir); end

poolobj = gcp('nocreate');
if isempty(poolobj), parpool('local', numWorkers); end

parfor k = 1:numel(files)
    rgb = double(imread(fullfile(files(k).folder, files(k).name)));
    if ndims(rgb) ~= 3 || size(rgb,3) ~= 3, error('Expected RGB TIFF: %s', files(k).name); end
    maxVal = max(rgb(:));
    if maxVal > 0, rgb = rgb / maxVal; end
    intensity = max(0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3), 1e-10);
    logImg = 1024 * log(intensity);
    lo = min(logImg(:)); hi = max(logImg(:));
    if hi - lo < 1e-12, logNorm = zeros(size(logImg));
    else, logNorm = (logImg - lo) / (hi - lo); end
    filtered = wls(logNorm, lambda, alpha);
    [~, name, ~] = fileparts(files(k).name);
    write_tiff(fullfile(outputInputDir, [name '-log.tiff']), logNorm);
    write_tiff(fullfile(outputLabelDir, [name '-WLS-' parameterTag '.tiff']), filtered);
end
fprintf('Done: %d pairs, lambda=%.3f, alpha=%.3f\n', numel(files), lambda, alpha);
fprintf('Input output: %s\nLabel output: %s\n', outputInputDir, outputLabelDir);

function write_tiff(fileName, img)
    t = Tiff(fileName, 'w');
    tags = struct('ImageLength',size(img,1),'ImageWidth',size(img,2), ...
        'SampleFormat',Tiff.SampleFormat.IEEEFP,'Photometric',Tiff.Photometric.MinIsBlack, ...
        'BitsPerSample',32,'SamplesPerPixel',1,'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky, ...
        'Compression',Tiff.Compression.None);
    t.setTag(tags); t.write(single(img)); t.close();
end

function ensure_dir(pathstr)
    if ~exist(pathstr,'dir'), mkdir(pathstr); end
end
