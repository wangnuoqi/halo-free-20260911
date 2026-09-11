function test_vivo_error_heatmaps(datasetName)

if nargin < 1 || isempty(datasetName)
    datasetName = 'test_vivo';
end

% Pixel-wise RGB error visualization for model comparisons.
% Red/yellow means larger absolute error. In the improvement map,
% improvement = baseline error - Exp7C error: blue means baseline is better
% (Exp7C is worse), while red means Exp7C is better.

thisDir = fileparts(mfilename('fullpath'));
modelRoot = fullfile(thisDir, datasetName);
tags = {'wls_l0p250_a1p200', 'wls_l0p500_a1p400', 'wls_l1p000_a1p600'};

for d = 1:numel(tags)
    inDir = fullfile(modelRoot, tags{d});
    outDir = fullfile(inDir, 'heatmaps');
    if ~exist(inDir, 'dir'), warning('Missing directory: %s', inDir); continue; end
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    gtFiles = dir(fullfile(inDir, '*_gt.jpg'));
    if isempty(gtFiles), warning('No *_gt.jpg files in %s', inDir); continue; end

    for k = 1:numel(gtFiles)
        [~, name, ~] = fileparts(gtFiles(k).name);
        sample = erase(name, '_gt');
        basePath = fullfile(inDir, [sample '_baseline.jpg']);
        expPath = fullfile(inDir, [sample '_exp7c.jpg']);
        if ~exist(basePath, 'file') || ~exist(expPath, 'file'), continue; end

        gt = im2double(imread(fullfile(inDir, gtFiles(k).name)));
        base = im2double(imread(basePath));
        exp7 = im2double(imread(expPath));
        h = min([size(gt,1), size(base,1), size(exp7,1)]);
        w = min([size(gt,2), size(base,2), size(exp7,2)]);
        gt = gt(1:h,1:w,:); base = base(1:h,1:w,:); exp7 = exp7(1:h,1:w,:);
        baseErr = mean(abs(base - gt), 3);
        expErr = mean(abs(exp7 - gt), 3);
        improvement = baseErr - expErr;
        % Use one shared range inside this sample's three-panel comparison.
        % Baseline and Exp7C are measured against the same GT in sRGB [0, 1].
        % Different samples may have different ranges, which is intentional:
        % the comparison is within one image, not across unrelated images.
        errMax = max([baseErr(:); expErr(:); eps]);
        impMax = max(abs(improvement(:))); if impMax < eps, impMax = 1; end

        fig = figure('Visible','off','Color','w','Position',[50 50 1900 700]);
        tiled = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
        ax = nexttile(tiled); imagesc(ax,baseErr,[0 errMax]); axis(ax,'image','off'); colormap(ax,hot); colorbar(ax); title(ax,'Baseline |RGB-GT|');
        ax = nexttile(tiled); imagesc(ax,expErr,[0 errMax]); axis(ax,'image','off'); colormap(ax,hot); colorbar(ax); title(ax,'Exp7C |RGB-GT|');
        ax = nexttile(tiled); imagesc(ax,improvement,[-impMax impMax]); axis(ax,'image','off'); colormap(ax,bluewhitered()); colorbar(ax); title(ax,'Baseline - Exp7C (blue=baseline better)');
        exportgraphics(fig,fullfile(outDir,[sample '_error_heatmap.jpg']),'Resolution',150);
        close(fig);

        baseGray = uint8(min(max(baseErr / errMax, 0), 1) * 255);
        expGray = uint8(min(max(expErr / errMax, 0), 1) * 255);
        imwrite(baseGray, fullfile(outDir,[sample '_baseline_error_gray.png']));
        imwrite(expGray, fullfile(outDir,[sample '_exp7c_error_gray.png']));
        fid = fopen(fullfile(outDir, [sample '_error_range.txt']), 'w');
        fprintf(fid, 'sample_shared_error_range=[0, %.9g]\n', errMax);
        fprintf(fid, 'sample_shared_improvement_range=[-%.9g, %.9g]\n', impMax, impMax);
        fprintf(fid, 'baseline_mean_error=%.9g\n', mean(baseErr(:)));
        fprintf(fid, 'exp7c_mean_error=%.9g\n', mean(expErr(:)));
        fprintf(fid, 'mean_improvement_baseline_minus_exp7c=%.9g\n', mean(improvement(:)));
        fprintf(fid, 'fraction_baseline_better=%.9g\n', mean(improvement(:) < 0));
        fprintf(fid, 'fraction_exp7c_better=%.9g\n', mean(improvement(:) > 0));
        fclose(fid);
    end
    fprintf('Saved heatmaps to %s\n', outDir);
end

end

function cmap = bluewhitered()
    n = 256; half = n/2;
    blue = [linspace(0,1,half)' linspace(0,1,half)' ones(half,1)];
    red = [ones(half,1) linspace(1,0,half)' linspace(1,0,half)'];
    cmap = [blue; red];
end
