clear; clc;

% Convert inference base TIFFs to RGB display images using the existing HDR
% detail-reconstruction logic.  Each sample is saved as one 3-panel figure:
% Original baseline | Exp7C recipe | WLS GT.

root = fileparts(mfilename('fullpath'));
edgeRoot = fileparts(root);
addpath(root);
dataRoot = 'D:\nuoqi\S2R-HDR\data\mixed_lavel_singlehdr';
hdrRoot = 'D:\nuoqi\S2R-HDR\data\alldata_origin\HDR_gt_sihdr';
inputRoot = fullfile(edgeRoot, 'outputs', 'mixed_lavel_singlehdr');
outputRoot = fullfile(edgeRoot, 'visualization', 'mixed_lavel_singlehdr_rgb_scale0p5');
% Linear-HDR display exposure multiplier.  This is visualization-only: it
% does not change inference outputs, WLS parameters, or PSNR/MSE metrics.
display_scale = 0.5;
mkdir_if_missing(outputRoot);

% The labels carry the per-sample WLS lambda/alpha.  The parser below accepts
% both the current NormalizedHDR naming and the older WLS naming.
gtFiles = dir(fullfile(dataRoot, 'testlabel', '*.tif*'));
for k = 1:numel(gtFiles)
    labelName = gtFiles(k).name;
    sampleKey = regexp(labelName, '^([0-9]+)', 'tokens', 'once');
    if isempty(sampleKey), continue; end
    sampleKey = sampleKey{1};
    [lambdaVal, alphaVal] = parse_wls_params(labelName);
    hdrPath = fullfile(hdrRoot, [sampleKey '.hdr']);
    if ~exist(hdrPath, 'file')
        warning('Missing HDR input: %s', hdrPath); continue;
    end
    img = hdrread(hdrPath);
    img = double(img);
    maxVal = max(img(:));
    if maxVal > 0, img = img / maxVal; end
    intensity = max(0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3), 1e-10);
    logIntensity = log(intensity);
    [logNorm, yMin, yMax] = nor_minmax(logIntensity);
    rangeVal = (yMax - yMin) + 1e-12;
    comp = 9937 / max(log(maxVal) * 1024, 1e-12);
    baseGt = yMin + wls(logNorm, lambdaVal, alphaVal) * rangeVal;
    minB = prctile(baseGt(:), 1);
    gtRgb = reconstruct_rgb(img, intensity, baseGt, comp, minB, display_scale);
    base = read_base(fullfile(inputRoot, 'baseline_original', [sampleKey '.tiff']));
    improved = read_base(fullfile(inputRoot, 'exp7c_recipe', [sampleKey '.tiff']));
    baselineRgb = reconstruct_rgb(img, intensity, yMin + resize_base(base, size(logIntensity)) * rangeVal, comp, minB, display_scale);
    improvedRgb = reconstruct_rgb(img, intensity, yMin + resize_base(improved, size(logIntensity)) * rangeVal, comp, minB, display_scale);
    figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1500 520]);
    subplot(1,3,1); imshow(baselineRgb); title('Original baseline');
    subplot(1,3,2); imshow(improvedRgb); title('Exp7C recipe');
    subplot(1,3,3); imshow(gtRgb); title(sprintf('WLS GT (lambda=%.4f, alpha=%.4f)', lambdaVal, alphaVal));
    exportgraphics(gcf, fullfile(outputRoot, sprintf('%s_compare.jpg', sampleKey)), 'Resolution', 150);
    close(gcf);
end
disp(['Saved RGB comparisons to: ' outputRoot]);

function [lambdaVal, alphaVal] = parse_wls_params(name)
    tok = regexp(name, 'wls-l([0-9eE+\-.]+)-a([0-9eE+\-.]+)', 'tokens', 'once');
    if isempty(tok), tok = regexp(name, 'WLS-([0-9eE+\-.]+)-a([0-9eE+\-.]+)', 'tokens', 'once'); end
    if isempty(tok)
        % Legacy mixed_lavel_singlehdr labels use WLS-025000, matching the
        % original MATLAB visualization convention lambda=.25, alpha=1.2.
        if ~isempty(regexp(name, 'WLS-025000', 'once'))
            lambdaVal = 0.25; alphaVal = 1.2;
        else
            error('Cannot parse WLS parameters from label: %s', name);
        end
    else
        lambdaVal = str2double(tok{1}); alphaVal = str2double(tok{2});
    end
end

function x = read_base(path)
    if ~exist(path, 'file'), error('Missing inference output: %s', path); end
    x = im2single(imread(path));
    if ndims(x) == 3, x = rgb2gray(x); end
end

function x = resize_base(x, targetSize)
    if ~isequal(size(x), targetSize), x = imresize(x, targetSize, 'bilinear'); end
end

function rgb = reconstruct_rgb(img, intensity, baseLog, comp, minB, display_scale)
    detailLog = log(intensity) - baseLog;
    targetLog = baseLog * comp + detailLog - (5.4 + minB * comp);
    ratio = exp(targetLog) ./ (intensity + 1e-10);
    rgb = min(max(img .* ratio, 0), 1);
    rgb = min(max(rgb * display_scale, 0), 1);
    rgb = srgb_gamma_correction(rgb);
end

function out = srgb_gamma_correction(in)
    out = zeros(size(in)); mask = in <= 0.0031308;
    out(mask) = 12.92 * in(mask);
    out(~mask) = 1.055 * in(~mask).^(1/2.4) - 0.055;
end

function [yn, ymin, ymax] = nor_minmax(y)
    ymin = min(y(:)); ymax = max(y(:)); yn = (y-ymin) / ((ymax-ymin)+1e-12);
end

function mkdir_if_missing(pathstr)
    if ~exist(pathstr, 'dir'), mkdir(pathstr); end
end
