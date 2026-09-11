clear; clc;

% Visualize the three test_vivo parameter protocols.  Model outputs are
% read from outputs/test_vivo/wls_* and matched to the original RGB TIFF by
% sample name.  The 9937/log(max)*1024 compression rule is retained for
% this device-normalized dataset.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

modelRoot = 'D:\nuoqi\halo-free\Decoupled_Learning\20260901_vis\edge\outputs\test_vivo';
inputDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
outputRoot = 'D:\nuoqi\halo-free\Decoupled_Learning\20260901_vis\edge\visualization\test_vivo';
display_scale = 1.0;
epsilon = 1e-10;
comp_numerator = 9937;

ensure_dir(outputRoot);
wlsDirs = dir(fullfile(modelRoot, 'wls_*'));
wlsDirs = wlsDirs([wlsDirs.isdir]);
inputFiles = [dir(fullfile(inputDir, '*.tiff')); dir(fullfile(inputDir, '*.tif'))];
if isempty(wlsDirs), error('No wls_* output directories found: %s', modelRoot); end
if isempty(inputFiles), error('No RGB TIFF inputs found: %s', inputDir); end

for d = 1:numel(wlsDirs)
    tag = wlsDirs(d).name;
    [lambdaVal, alphaVal] = parse_wls_dir(tag);
    root = fullfile(modelRoot, tag);
    baseDir = fullfile(root, 'baseline_original');
    expDir = fullfile(root, 'exp7c_recipe');
    outDir = fullfile(outputRoot, tag);
    ensure_dir(outDir);
    fprintf('\n=== %s: lambda=%.4f alpha=%.4f ===\n', tag, lambdaVal, alphaVal);

    for k = 1:numel(inputFiles)
        [~, name, ~] = fileparts(inputFiles(k).name);
        rgb = double(imread(fullfile(inputFiles(k).folder, inputFiles(k).name)));
        if ndims(rgb) ~= 3 || size(rgb,3) ~= 3, error('Input is not RGB: %s', inputFiles(k).name); end
        maxVal = max(rgb(:));
        if maxVal > 0, rgb = rgb / maxVal; end
        intensity = max(0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3), epsilon);
        logIntensity = log(intensity);
        [logNorm, yMin, yMax] = nor_minmax(logIntensity);
        rangeVal = (yMax-yMin) + epsilon;
        comp = comp_numerator / max(log(maxVal)*1024, epsilon);
        wlsBase = yMin + wls(logNorm, lambdaVal, alphaVal) * rangeVal;
        minB = prctile(wlsBase(:), 1);

        base = yMin + resize_to(read_gray(fullfile(baseDir, [name '.tiff'])), size(logIntensity)) * rangeVal;
        exp7 = yMin + resize_to(read_gray(fullfile(expDir, [name '.tiff'])), size(logIntensity)) * rangeVal;
        rgbBase = reconstruct_rgb(rgb, intensity, base, comp, minB, display_scale);
        rgbExp = reconstruct_rgb(rgb, intensity, exp7, comp, minB, display_scale);
        rgbGt = reconstruct_rgb(rgb, intensity, wlsBase, comp, minB, display_scale);

        imwrite(uint8(rgbBase*255), fullfile(outDir, [name '_baseline.jpg']));
        imwrite(uint8(rgbExp*255), fullfile(outDir, [name '_exp7c.jpg']));
        imwrite(uint8(rgbGt*255), fullfile(outDir, [name '_gt.jpg']));
        fig = figure('Visible','off','Color','w','Position',[50 50 1900 680]);
        tiled = tiledlayout(fig,1,3,'TileSpacing','none','Padding','compact');
        ax = nexttile(tiled); imshow(rgbBase,'Parent',ax); title(ax,'Baseline refine');
        ax = nexttile(tiled); imshow(rgbExp,'Parent',ax); title(ax,'Exp7C refine');
        ax = nexttile(tiled); imshow(rgbGt,'Parent',ax); title(ax,sprintf('WLS GT (%.2f, %.2f)',lambdaVal,alphaVal));
        exportgraphics(fig,fullfile(outDir,[name '_compare.jpg']),'Resolution',150);
        close(fig);
    end
    fprintf('Saved %d samples to %s\n', numel(inputFiles), outDir);
end

function [lambdaVal, alphaVal] = parse_wls_dir(name)
    t = regexp(name,'^wls_l([0-9]+)p([0-9]+)_a([0-9]+)p([0-9]+)$','tokens','once');
    if isempty(t), error('Cannot parse WLS directory: %s',name); end
    lambdaVal = str2double([t{1} '.' t{2}]); alphaVal = str2double([t{3} '.' t{4}]);
end

function x = read_gray(pathstr)
    if ~exist(pathstr,'file'), error('Missing model output: %s',pathstr); end
    x = im2single(imread(pathstr));
    if ndims(x)==3, x=rgb2gray(x); end
end

function x = resize_to(x,targetSize)
    if ~isequal(size(x),targetSize), x=imresize(x,targetSize,'bilinear'); end
end

function rgb = reconstruct_rgb(img,intensity,baseLog,comp,minB,scale)
    detailLog = log(intensity)-baseLog;
    targetLog = baseLog*comp + detailLog - (5.4 + minB*comp);
    ratio = exp(targetLog)./(intensity+1e-10);
    rgb = min(max(img.*ratio,0),1);
    rgb = srgb_gamma(min(max(rgb*scale,0),1));
end

function out = srgb_gamma(in)
    out=zeros(size(in)); mask=in<=0.0031308;
    out(mask)=12.92*in(mask); out(~mask)=1.055*in(~mask).^(1/2.4)-0.055;
end

function [yn,ymin,ymax] = nor_minmax(y)
    ymin=min(y(:)); ymax=max(y(:)); yn=(y-ymin)/((ymax-ymin)+1e-12);
end

function ensure_dir(pathstr)
    if ~exist(pathstr,'dir'), mkdir(pathstr); end
end
