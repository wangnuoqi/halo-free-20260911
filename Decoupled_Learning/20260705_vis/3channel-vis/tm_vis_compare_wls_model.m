function tm_vis_compare_wls_model()
% Compare tone-mapped RGB visualizations reconstructed from:
% 1) WLS base estimated from the normalized log input
% 2) External model base loaded from float TIFF/PNG/MAT/EXR
%
% Default behavior:
% - reconstructs base back to the restored 1024*log domain
% - computes detail in the restored log domain
% - compresses base in the linear domain using a p95-anchored GGIF-style curve

cfg = default_config();
ensure_dir(cfg.output_root);
ensure_dir(fullfile(cfg.output_root, 'wls_rgb'));
ensure_dir(fullfile(cfg.output_root, 'model_rgb'));
ensure_dir(fullfile(cfg.output_root, 'panels'));

input_files = collect_files(cfg.input_dir, cfg.input_exts);
wls_files = collect_files(cfg.wls_dir, cfg.wls_exts);
model_files = collect_files(cfg.model_dir, cfg.model_exts);
matched = match_by_sample_stem(input_files, wls_files, model_files);
if isempty(matched)
    error('No matched input/WLS/model file triplets found.');
end

fprintf('Found %d matched pairs.\n', numel(matched));

for k = 1:numel(matched)
    input_path = matched(k).input_path;
    wls_path = matched(k).wls_path;
    model_path = matched(k).model_path;
    sample_id = matched(k).sample_id;

    img_rgb = read_input_rgb(input_path);
    if size(img_rgb, 3) ~= 3
        fprintf('Skip %s because it is not 3-channel.\n', input_path);
        continue;
    end

    [img_work, prep_info] = preprocess_input_rgb(img_rgb, cfg);
    intensity = compute_intensity(img_work);
    log_input = cfg.log_scale * log(intensity + cfg.epsilon);
    [log_input_norm, y_min, y_max] = nor_minmax(log_input);

    wls_base_norm = read_base_map(wls_path);
    if ~isequal(size(wls_base_norm), size(log_input_norm))
        wls_base_norm = imresize(wls_base_norm, size(log_input_norm), 'bilinear');
    end
    model_base_norm = read_model_filtered(model_path);
    if ~isequal(size(model_base_norm), size(log_input_norm))
        model_base_norm = imresize(model_base_norm, size(log_input_norm), 'bilinear');
    end

    wls_pack = reconstruct_from_base(img_work, intensity, log_input, wls_base_norm, y_min, y_max, cfg);
    model_pack = reconstruct_from_base(img_work, intensity, log_input, model_base_norm, y_min, y_max, cfg);

    wls_out_path = fullfile(cfg.output_root, 'wls_rgb', [sample_id, '_wls.jpg']);
    model_out_path = fullfile(cfg.output_root, 'model_rgb', [sample_id, '_model.jpg']);
    imwrite(wls_pack.img_u8, wls_out_path);
    imwrite(model_pack.img_u8, model_out_path);

    panel_path = fullfile(cfg.output_root, 'panels', [sample_id, '_panel.jpg']);
    save_panel(panel_path, sample_id, img_work, prep_info, wls_pack, model_pack);

    fprintf('Processed %s\n', sample_id);
end

fprintf('All results saved to %s\n', cfg.output_root);
end

function cfg = default_config()
cfg = struct();
cfg.input_dir = 'D:\nuoqi\S2R-HDR\data\test_fei\HDRimage';
cfg.wls_dir = 'D:\nuoqi\S2R-HDR\data\test_fei\testlabel';
cfg.model_dir = 'D:\nuoqi\S2R-HDR\data\test_fei\rf3stage_output';
cfg.output_root = 'D:\nuoqi\halo-free\Decoupled_Learning\20260705_vis\3channel-vis\results_test_fei_rf3stage';
cfg.input_exts = {'.hdr', '.exr', '.mat', '.tiff', '.tif', '.png'};
cfg.wls_exts = {'.tiff', '.tif', '.png', '.exr', '.mat'};
cfg.model_exts = {'.tiff', '.tif', '.png', '.exr', '.mat'};

cfg.epsilon = 1e-10;
cfg.log_scale = 1024.0;
cfg.wls_lambda = 0.25;
cfg.wls_alpha = 1.2;

cfg.input_scale_mode = 'global_max';
cfg.fixed_input_peak = 1.0;

cfg.compress_percentile = 95;
cfg.compress_target = 0.8;
end

function ensure_dir(path_str)
if ~exist(path_str, 'dir')
    mkdir(path_str);
end
end

function files = collect_files(root_dir, exts)
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(root_dir, ['*', exts{i}]))]; %#ok<AGROW>
end
if isempty(files)
    error('No files found in %s', root_dir);
end
[~, order] = sort(lower({files.name}));
files = files(order);
end

function matched = match_by_sample_stem(input_files, wls_files, model_files)
input_map = containers.Map();
for i = 1:numel(input_files)
    sid = sample_stem_from_name(input_files(i).name);
    if ~isempty(sid)
        input_map(sid) = fullfile(input_files(i).folder, input_files(i).name);
    end
end

wls_map = containers.Map();
for i = 1:numel(wls_files)
    sid = sample_stem_from_name(wls_files(i).name);
    if ~isempty(sid)
        wls_map(sid) = fullfile(wls_files(i).folder, wls_files(i).name);
    end
end

matched = struct('sample_id', {}, 'input_path', {}, 'wls_path', {}, 'model_path', {});
for i = 1:numel(model_files)
    sid = sample_stem_from_name(model_files(i).name);
    if isempty(sid) || ~isKey(input_map, sid) || ~isKey(wls_map, sid)
        continue;
    end
    entry.sample_id = sid; %#ok<STRNU>
    entry.input_path = input_map(sid);
    entry.wls_path = wls_map(sid);
    entry.model_path = fullfile(model_files(i).folder, model_files(i).name);
    matched(end + 1) = entry; %#ok<AGROW>
end
end

function sid = sample_stem_from_name(name)
[~, stem, ~] = fileparts(name);
sid = regexprep(stem, '(-WLS-\d+|-log_pred|-log)$', '');
end

function img = read_input_rgb(path_str)
[~, ~, ext] = fileparts(path_str);
ext = lower(ext);

switch ext
    case '.hdr'
        img = double(hdrread(path_str));
    case '.exr'
        img = double(exrread(path_str));
    case {'.tif', '.tiff', '.png'}
        img = double(imread(path_str));
    case '.mat'
        data = load(path_str);
        if isfield(data, 'img')
            img = double(data.img);
        else
            error('MAT input missing img field: %s', path_str);
        end
    otherwise
        error('Unsupported input extension: %s', ext);
end

if size(img, 3) == 1
    img = repmat(img, [1, 1, 3]);
end
end

function img_filtered = read_model_filtered(path_str)
[~, ~, ext] = fileparts(path_str);
ext = lower(ext);

switch ext
    case {'.tif', '.tiff', '.png'}
        x = im2single(imread(path_str));
        if ndims(x) == 3
            x = rgb2gray(x);
        end
        img_filtered = single(x);
    case '.exr'
        x = single(exrread(path_str));
        if ndims(x) == 3
            x = rgb2gray(x);
        end
        img_filtered = x;
    case '.mat'
        data = load(path_str);
        cand = {'img_filtered', 'pred', 'output', 'y', 'base'};
        found = '';
        for i = 1:numel(cand)
            if isfield(data, cand{i})
                found = cand{i};
                break;
            end
        end
        if isempty(found)
            error('MAT model output missing known field: %s', path_str);
        end
        x = single(data.(found));
        if ndims(x) == 3
            x = rgb2gray(x);
        end
        img_filtered = x;
    otherwise
        error('Unsupported model extension: %s', ext);
end
end

function img_filtered = read_base_map(path_str)
img_filtered = read_model_filtered(path_str);
end

function [img_work, prep_info] = preprocess_input_rgb(img_rgb, cfg)
prep_info = struct();
prep_info.input_min = min(img_rgb(:));
prep_info.input_max = max(img_rgb(:));

switch cfg.input_scale_mode
    case 'global_max'
        peak = max(img_rgb(:));
    case 'fixed_peak'
        peak = cfg.fixed_input_peak;
    otherwise
        error('Unsupported input_scale_mode: %s', cfg.input_scale_mode);
end

if peak <= 0
    peak = 1.0;
end

img_work = img_rgb / peak;
prep_info.scale_peak = peak;
prep_info.output_min = min(img_work(:));
prep_info.output_max = max(img_work(:));
end

function intensity = compute_intensity(img_rgb)
intensity = 0.299 * img_rgb(:, :, 1) + 0.587 * img_rgb(:, :, 2) + 0.114 * img_rgb(:, :, 3);
end

function [y_norm, y_min, y_max] = nor_minmax(y)
y_min = min(y(:));
y_max = max(y(:));
rangev = (y_max - y_min) + 1e-12;
y_norm = (y - y_min) / rangev;
end

function pack = reconstruct_from_base(img_rgb, intensity, log_input, base_norm, y_min, y_max, cfg)
base_log = y_min + double(base_norm) * ((y_max - y_min) + 1e-12);
detail_log = log_input - base_log;

base_linear = exp(base_log / cfg.log_scale);
ref = prctile(base_linear(:), cfg.compress_percentile);
ref = max(ref, cfg.epsilon);

target = cfg.compress_target;
target = min(max(target, 1e-4), 1 - 1e-4);
gain = target / (1.0 - target);

x = base_linear / ref;
compressed_base_linear = (gain * x) ./ (1.0 + gain * x);
compressed_base_linear = max(compressed_base_linear, cfg.epsilon);
compressed_base_log = cfg.log_scale * log(compressed_base_linear);

target_log = compressed_base_log + detail_log;
tone_linear = exp(target_log / cfg.log_scale);
ratio = tone_linear ./ (intensity + cfg.epsilon);
img_out = img_rgb .* ratio;
img_out = min(max(img_out, 0), 1);
img_out_srgb = srgb_gamma_correction(img_out);
img_u8 = uint8(min(max(img_out_srgb, 0), 1) * 255);

pack = struct();
pack.base_norm = base_norm;
pack.base_log = base_log;
pack.detail_log = detail_log;
pack.compressed_base_linear = compressed_base_linear;
pack.img_out = img_out;
pack.img_u8 = img_u8;
pack.ref = ref;
pack.target = target;
end

function out = srgb_gamma_correction(in)
out = zeros(size(in), 'like', in);
mask = in <= 0.0031308;
out(mask) = 12.92 * in(mask);
out(~mask) = 1.055 * (in(~mask) .^ (1 / 2.4)) - 0.055;
end

function save_panel(path_str, sample_id, img_work, prep_info, wls_pack, model_pack)
fig = figure('Visible', 'off', 'Position', [100, 100, 1800, 900]);
tiledlayout(2, 4, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imshow(min(max(img_work, 0), 1));
title(sprintf('Input RGB (peak=%.4g)', prep_info.scale_peak), 'Interpreter', 'none');

nexttile;
imagesc(wls_pack.base_norm);
axis image off;
colormap(gca, gray);
colorbar;
title(sprintf('WLS base norm, p95=%.4f', wls_pack.ref), 'Interpreter', 'none');

nexttile;
imagesc(model_pack.base_norm);
axis image off;
colormap(gca, gray);
colorbar;
title(sprintf('Model base norm, p95=%.4f', model_pack.ref), 'Interpreter', 'none');

nexttile;
imagesc(abs(double(wls_pack.base_norm) - double(model_pack.base_norm)));
axis image off;
colormap(gca, hot);
colorbar;
title('|WLS norm - Model norm|');

nexttile;
imagesc(wls_pack.detail_log);
axis image off;
colormap(gca, parula);
colorbar;
title('WLS detail in restored 1024-log');

nexttile;
imagesc(model_pack.detail_log);
axis image off;
colormap(gca, parula);
colorbar;
title('Model detail in restored 1024-log');

nexttile;
imshow(wls_pack.img_u8);
title('WLS tone-mapped RGB');

nexttile;
imshow(model_pack.img_u8);
title('Model tone-mapped RGB');

sgtitle(sprintf('%s | restored-log detail + p95-anchored base compression', sample_id), 'Interpreter', 'none');
exportgraphics(fig, path_str, 'Resolution', 160);
close(fig);
end

function out = run_wls_filter(in, lambda, alpha)
L = in;
small_num = 1e-4;
[r, c] = size(in);
k = r * c;

dy = diff(L, 1, 1);
dy = -lambda ./ (abs(dy) .^ alpha + small_num);
dy = [dy; zeros(1, c)];
dy = dy(:);

dx = diff(L, 1, 2);
dx = -lambda ./ (abs(dx) .^ alpha + small_num);
dx = [dx, zeros(r, 1)];
dx = dx(:);

B = zeros(k, 2);
B(:, 1) = dx;
B(:, 2) = dy;
A = spdiags(B, [-r, -1], k, k);

e = dx;
w = [zeros(r, 1); dx(1:end-r)];
s = dy;
n = [0; dy(1:end-1)];
D = 1 - (e + w + s + n);
A = A + A' + spdiags(D, 0, k, k);

out = A \ in(:);
out = reshape(out, r, c);
end
