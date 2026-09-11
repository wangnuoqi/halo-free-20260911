clear; clc;

% Visualize test_fei outputs produced by run_inference_compare.py.
% Each WLS directory contains baseline_original, exp7c_recipe and gt.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

modelRoot = 'D:\nuoqi\halo-free\Decoupled_Learning\20260901_vis\edge\outputs\test_fei';
inputDir = 'D:\nuoqi\S2R-HDR\data\test_fei\HDRimage';
outputRoot = 'D:\nuoqi\halo-free\Decoupled_Learning\20260901_vis\edge\visualization\test_fei';
% Direct tone-mapping compression factor for linear test_fei HDR.
% Set this explicitly because test_fei has no device-limited 9937 code range.
compression_factor_override = 0.45;
display_scale = 1.0;
epsilon = 1e-10;
comp_numerator = 9937;

ensure_dir(outputRoot);
wlsDirs = dir(fullfile(modelRoot, 'wls_*'));
wlsDirs = wlsDirs([wlsDirs.isdir]);
if isempty(wlsDirs), error('No wls_* directories found under %s', modelRoot); end

inputFiles = dir(fullfile(inputDir, '*.hdr'));
if isempty(inputFiles), error('No HDR files found under %s', inputDir); end

for d = 1:numel(wlsDirs)
    paramTag = wlsDirs(d).name;
    [lambdaVal, alphaVal] = parse_wls_dir(paramTag);
    paramRoot = fullfile(modelRoot, paramTag);
    baselineDir = fullfile(paramRoot, 'baseline_original');
    improvedDir = fullfile(paramRoot, 'exp7c_recipe');
    gtDir = fullfile(paramRoot, 'gt');
    outDir = fullfile(outputRoot, paramTag);
    ensure_dir(outDir);
    fprintf('\n=== %s (lambda=%.4f, alpha=%.4f) ===\n', paramTag, lambdaVal, alphaVal);

    for k = 1:numel(inputFiles)
        [~, sampleName, ~] = fileparts(inputFiles(k).name);
        hdr = double(hdrread(fullfile(inputFiles(k).folder, inputFiles(k).name)));
        maxVal = max(hdr(:));
        if maxVal > 0, hdr = hdr / maxVal; end
        intensity = max(0.299 * hdr(:,:,1) + 0.587 * hdr(:,:,2) + 0.114 * hdr(:,:,3), epsilon);
        logIntensity = log(intensity);
        [logNorm, yMin, yMax] = nor_minmax(logIntensity);
        rangeVal = (yMax - yMin) + epsilon;
        if isempty(compression_factor_override)
            comp = comp_numerator / max(log(maxVal) * 1024, epsilon);
        else
            comp = compression_factor_override;
        end
        wlsBase = yMin + wls(logNorm, lambdaVal, alphaVal) * rangeVal;
        minB = prctile(wlsBase(:), 1);

        baseline = read_gray(fullfile(baselineDir, [sampleName '.tiff']));
        improved = read_gray(fullfile(improvedDir, [sampleName '.tiff']));
        baseline = yMin + resize_to(baseline, size(logIntensity)) * rangeVal;
        improved = yMin + resize_to(improved, size(logIntensity)) * rangeVal;

        rgbBase = reconstruct_rgb(hdr, intensity, baseline, comp, minB, display_scale);
        rgbImproved = reconstruct_rgb(hdr, intensity, improved, comp, minB, display_scale);
        rgbGt = reconstruct_rgb(hdr, intensity, wlsBase, comp, minB, display_scale);

        % Save the individual color images alongside the three-panel figure.
        imwrite(uint8(rgbBase * 255), fullfile(outDir, [sampleName '_baseline.jpg']));
        imwrite(uint8(rgbImproved * 255), fullfile(outDir, [sampleName '_exp7c.jpg']));
        imwrite(uint8(rgbGt * 255), fullfile(outDir, [sampleName '_gt.jpg']));

        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1900 680]);
        tiled = tiledlayout(fig, 1, 3, 'TileSpacing', 'none', 'Padding', 'compact');
        ax = nexttile(tiled); imshow(rgbBase, 'Parent', ax); title(ax, 'Baseline refine');
        ax = nexttile(tiled); imshow(rgbImproved, 'Parent', ax); title(ax, 'Exp7C refine');
        ax = nexttile(tiled); imshow(rgbGt, 'Parent', ax); title(ax, sprintf('WLS GT (%.2f, %.2f)', lambdaVal, alphaVal));
        exportgraphics(fig, fullfile(outDir, [sampleName '_compare.jpg']), 'Resolution', 150);
        close(fig);
    end
    fprintf('Saved %d comparisons to %s\n', numel(inputFiles), outDir);
end

function [lambdaVal, alphaVal] = parse_wls_dir(name)
    token = regexp(name, '^wls_l([0-9]+)p([0-9]+)_a([0-9]+)p([0-9]+)$', 'tokens', 'once');
    if isempty(token), error('Cannot parse WLS directory name: %s', name); end
    lambdaVal = str2double([token{1} '.' token{2}]);
    alphaVal = str2double([token{3} '.' token{4}]);
end

function x = read_gray(pathstr)
    if ~exist(pathstr, 'file'), error('Missing model output: %s', pathstr); end
    x = im2single(imread(pathstr));
    if ndims(x) == 3, x = rgb2gray(x); end
end

function x = resize_to(x, targetSize)
    if ~isequal(size(x), targetSize), x = imresize(x, targetSize, 'bilinear'); end
end

function rgb = reconstruct_rgb(img, intensity, baseLog, comp, minB, displayScale)
    detailLog = log(intensity) - baseLog;
    targetLog = baseLog * comp + detailLog - (5.4 + minB * comp);
    ratio = exp(targetLog) ./ (intensity + 1e-10);
    rgb = min(max(img .* ratio, 0), 1);
    rgb = min(max(rgb * displayScale, 0), 1);
    rgb = srgb_gamma(rgb);
end

function out = srgb_gamma(in)
    out = zeros(size(in));
    mask = in <= 0.0031308;
    out(mask) = 12.92 * in(mask);
    out(~mask) = 1.055 * in(~mask).^(1/2.4) - 0.055;
end

function [yn, ymin, ymax] = nor_minmax(y)
    ymin = min(y(:)); ymax = max(y(:));
    yn = (y - ymin) / ((ymax - ymin) + 1e-12);
end

function ensure_dir(pathstr)
    if ~exist(pathstr, 'dir'), mkdir(pathstr); end
end
