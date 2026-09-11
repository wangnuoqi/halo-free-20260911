clear all;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

inputPath = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
baseOutputPath = 'D:\nuoqi\S2R-HDR\data\test_vivo\params';
labelOutputPath = fullfile(baseOutputPath, 'testlabel');
inputOutputPath = fullfile(baseOutputPath, 'testinput');

files = dir(fullfile(inputPath, '*.tiff'));
if isempty(files)
    files = dir(fullfile(inputPath, '*.tif'));
end

if isempty(files)
    error('No TIFF files found in inputPath: %s', inputPath);
end

numVariantsPerImage = 7;
lambdaBase = 0.25;
alphaBase = 1.2;
lambdaLogRange = [-log(10), log(10)];
alphaRange = [0.8, 1.6];
epsVal = 1e-10;
numWorkers = 32;

if ~exist(baseOutputPath, 'dir')
    mkdir(baseOutputPath);
end
if ~exist(labelOutputPath, 'dir')
    mkdir(labelOutputPath);
end
if ~exist(inputOutputPath, 'dir')
    mkdir(inputOutputPath);
end

poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end
parpool(numWorkers);

parfor k = 1:length(files)
    inputImagePath = fullfile(inputPath, files(k).name);
    img = double(imread(inputImagePath));

    maxVal = max(img(:));
    if maxVal > 0
        img = img / maxVal;
    end

    if ndims(img) == 3
        grayImg = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
    else
        grayImg = img;
    end
    grayImg = double(grayImg);

    linearImg = max(grayImg, epsVal);
    logImg = 1024 * log(linearImg);
    logImg = nor(logImg);

    [height, width] = size(logImg);
    [~, baseName, ~] = fileparts(files(k).name);
    ext = '.tiff';

    for m = 1:numVariantsPerImage
        lambdaRand = rand() * range(lambdaLogRange) + min(lambdaLogRange);
        lambda = exp(lambdaRand) * lambdaBase;
        alpha = rand() * range(alphaRange) + min(alphaRange);

        imgFiltered = wls(logImg, lambda, alpha);

        variantName = sprintf('%s-wls-l%.6f-a%.6f-h%d-w%d-v%d', ...
            baseName, lambda, alpha, height, width, m);

        labelFileName = fullfile(labelOutputPath, [variantName, ext]);
        inputFileName = fullfile(inputOutputPath, [variantName, ext]);

        write_single_channel_tiff(labelFileName, imgFiltered);
        write_single_channel_tiff(inputFileName, logImg);
    end

    fprintf('Saved %d pairs for %s\n', numVariantsPerImage, files(k).name);
end

labelFiles = dir(fullfile(labelOutputPath, '*.tiff'));
inputFiles = dir(fullfile(inputOutputPath, '*.tiff'));
fprintf('Done. label files: %d, input files: %d\n', length(labelFiles), length(inputFiles));
disp('所有图像已按参数化方式生成并保存完成！');

function write_single_channel_tiff(fileName, img)
    t = Tiff(fileName, 'w');
    tagstruct = struct();
    tagstruct.ImageLength = size(img, 1);
    tagstruct.ImageWidth = size(img, 2);
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;
    t.setTag(tagstruct);
    t.write(single(img));
    t.close();
end
