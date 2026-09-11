clear; clc;
% 多参数模型输出可视化：
% 1) 先对每个 HDR 输入跑一次固定 WLS，可视化并提取 minB
% 2) 再读取 modelOutDir 中每个参数化输出，按文件名里的 lambda/alpha 重建 tone-mapped vis
% 3) 所有可视化保存到 /mnt/d/nuoqi/S2R-HDR/data/test_vivo/params 下

%% ====== Path Configuration ======
modelOutDir = 'D:\nuoqi\halo-free\Decoupled-params\try-freezb\eval_outputs\test_vivo_params\v5_best';
inputDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
labelDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\params\testlabel';

outputRoot = 'D:\nuoqi\S2R-HDR\data\test_vivo\params';
outputPath_wls_rgb = fullfile(outputRoot, 'vis_wls_gt');
outputPath_model_rgb = fullfile(outputRoot, 'vis_model_params_v5_best');

ensure_dir(outputRoot);
ensure_dir(outputPath_wls_rgb);
ensure_dir(outputPath_model_rgb);
start_parallel_pool();

%% ====== Constants ======
epsilon = 1e-10;
comp_numerator = 9937;

%% ====== Collect and Sort Files ======
model_exts = {'*.tiff','*.tif','*.png','*.exr','*.mat'};
model_files = [];
for i = 1:numel(model_exts)
    model_files = [model_files; dir(fullfile(modelOutDir, model_exts{i}))]; %#ok<AGROW>
end
if isempty(model_files)
    error('No model output files found in: %s', modelOutDir);
end
model_names = {model_files.name};
[~, idx] = sort(model_names);
model_files = model_files(idx);

input_exts = {'*.hdr','*.exr','*.mat','*.tiff','*.tif','*.png'};
input_files = [];
for i = 1:numel(input_exts)
    input_files = [input_files; dir(fullfile(inputDir, input_exts{i}))]; %#ok<AGROW>
end
if isempty(input_files)
    error('No input files found in: %s', inputDir);
end
input_names = {input_files.name};
[~, idx] = sort(input_names);
input_files = input_files(idx);

label_files = dir(fullfile(labelDir, '*.tiff'));
if isempty(label_files)
    label_files = dir(fullfile(labelDir, '*.tif'));
end
if isempty(label_files)
    error('No label files found in: %s', labelDir);
end
label_names = {label_files.name};
[~, idx] = sort(label_names);
label_files = label_files(idx);

N = min([numel(model_files), numel(label_files)]);
if numel(label_files) ~= numel(model_files)
    error(['Label/model count mismatch detected:\n' ...
           '  Label files: %d\n' ...
           '  Model outputs: %d\n' ...
           'Please ensure the evaluation output count matches testlabel exactly.'], ...
          numel(label_files), numel(model_files));
end

%% ====== Build Input Index ======
input_keys = cell(numel(input_files), 1);
for i = 1:numel(input_files)
    input_keys{i} = get_input_key(input_files(i).name);
end

%% ====== Compute comp_values ======
comp_denominator_values = compute_comp_denominators_lnmax_times_1024(input_files);
comp_values = comp_numerator ./ comp_denominator_values;

fprintf('Auto-computed comp_values = 9937 ./ (ln(max(input)) * 1024):\n');
disp(comp_values');

%% ====== Model VIS Main Processing Loop ======
parfor k = 1:N
    fileModel = model_files(k).name;
    fileLabel = label_files(k).name;
    [sample_key, lambda_model, alpha_model, variant_tag] = parse_label_filename(fileLabel);

    match_idx = find(strcmp(input_keys, sample_key), 1);
    if isempty(match_idx)
        warning('No matching HDR input found for model output: %s', fileModel);
        continue;
    end

    inputPath = fullfile(input_files(match_idx).folder, input_files(match_idx).name);
    img = read_input_rgb(inputPath);

    if size(img, 3) ~= 3
        warning('Image %s is not 3-channel, skipping', input_files(match_idx).name);
        continue;
    end

    max_val = max(img(:));
    if max_val > 0
        img = img / max_val;
    end

    intensity = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
    intensity = max(intensity, epsilon);
    log_intensity = log(intensity);
    [log_intensity_nor, y_min, y_max] = nor_minmax(log_intensity);
    rangev = (y_max - y_min) + 1e-12;

    compression_factor = comp_values(match_idx);

    % 每个模型输出都用自身文件名里的 lambda/alpha 去生成对应 WLS GT，
    % 并从该参数对应的 base_log 中提取 minB。
    base_log_nor_gt = wls(log_intensity_nor, lambda_model, alpha_model);
    base_log_gt = y_min + base_log_nor_gt * rangev;
    minB = prctile(base_log_gt(:), 1);

    detail_log_gt = log_intensity - base_log_gt;
    target_log_gt = (base_log_gt * compression_factor) + detail_log_gt;
    target_log_gt = target_log_gt - (5.4 + minB * compression_factor);
    detailed_channel_gt = exp(target_log_gt);
    ratio_gt = detailed_channel_gt ./ (intensity + epsilon);
    img_out_gt = img .* ratio_gt;
    img_out_gt = min(max(img_out_gt, 0), 1);
    img_out_gt_srgb = srgb_gamma_correction(img_out_gt);
    img_u8_gt = uint8(img_out_gt_srgb * 255);

    sample_wls_out_dir = fullfile(outputPath_wls_rgb, sample_key);
    ensure_dir(sample_wls_out_dir);
    outputFileName_gt = fullfile(sample_wls_out_dir, ...
        sprintf('%s_%s_gt_l%.6f_a%.6f.jpg', sample_key, variant_tag, lambda_model, alpha_model));
    if ~exist(outputFileName_gt, 'file')
        imwrite(img_u8_gt, outputFileName_gt);
    end

    inPathFiltered = fullfile(model_files(k).folder, fileModel);
    img_filtered = read_model_filtered(inPathFiltered);

    base_log = y_min + img_filtered * rangev;

    if ~isequal(size(base_log), size(log_intensity))
        base_log = imresize(base_log, size(log_intensity), 'bilinear');
    end

    detail_log = log_intensity - base_log;
    target_log = (base_log * compression_factor) + detail_log;
    target_log = target_log - (5.4 + minB * compression_factor);

    detailed_channel = exp(target_log);
    ratio = detailed_channel ./ (intensity + epsilon);
    img_out = img .* ratio;

    img_out = min(max(img_out, 0), 1);
    img_out_srgb = srgb_gamma_correction(img_out);
    img_u8 = uint8(img_out_srgb * 255);

    sample_out_dir = fullfile(outputPath_model_rgb, sample_key);
    ensure_dir(sample_out_dir);

    outputFileName = fullfile(sample_out_dir, ...
        sprintf('%s_%s_l%.6f_a%.6f.jpg', sample_key, variant_tag, lambda_model, alpha_model));
    imwrite(img_u8, outputFileName);
end

disp('All parameterized model visualization images processed and saved.');

%% ====== Helper Functions ======
function key = get_input_key(filename)
    [~, name, ~] = fileparts(filename);
    key = regexprep(name, '_?NormalizedHDR$', '');
end

function [sample_key, lambda_val, alpha_val, variant_tag] = parse_label_filename(filename)
    [~, name, ~] = fileparts(filename);
    tokens = regexp(name, '^(.*)_NormalizedHDR-wls-l([0-9eE+\-.]+)-a([0-9eE+\-.]+)-h\d+-w\d+-(v\d+)$', ...
        'tokens', 'once');
    if isempty(tokens)
        error('Cannot parse label filename: %s', filename);
    end

    sample_key = tokens{1};
    lambda_val = str2double(tokens{2});
    alpha_val = str2double(tokens{3});
    variant_tag = tokens{4};
end

function ensure_dir(pathstr)
    if ~exist(pathstr, 'dir')
        mkdir(pathstr);
    end
end

function start_parallel_pool()
    numWorkers = 4;
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', numWorkers);
    elseif isa(pool, 'parallel.ThreadPool')
        delete(pool);
        pool = parpool('local', numWorkers);
    elseif pool.NumWorkers ~= numWorkers
        delete(pool);
        pool = parpool('local', numWorkers);
    end
    fprintf('Parallel pool started with %d workers\n', pool.NumWorkers);
end

function img_filtered = read_model_filtered(path)
    [~,~,ext] = fileparts(path);
    ext = lower(ext);

    switch ext
        case {'.tif','.tiff','.png'}
            x = imread(path);
            x = im2single(x);
            if ndims(x) == 3
                x = rgb2gray(x);
            end
            img_filtered = x;

        case '.exr'
            x = exrread(path);
            x = single(x);
            if ndims(x) == 3
                x = rgb2gray(x);
            end
            img_filtered = x;

        case '.mat'
            S = load(path);
            cand = {'img_filtered','pred','output','y','base'};
            found = '';
            for i = 1:numel(cand)
                if isfield(S, cand{i})
                    found = cand{i};
                    break;
                end
            end
            if isempty(found)
                error('Model output .mat file does not contain expected variables: %s', path);
            end
            x = S.(found);
            x = single(x);
            if ndims(x) == 3
                x = rgb2gray(x);
            end
            img_filtered = x;

        otherwise
            error('Unsupported model output format: %s', ext);
    end
end

function img = read_input_rgb(path)
    [~,~,ext] = fileparts(path);
    ext = lower(ext);

    switch ext
        case '.hdr'
            img = hdrread(path);
            img = double(img);

        case '.exr'
            img = exrread(path);
            img = double(img);

        case {'.tif','.tiff','.png'}
            img = double(imread(path));
            if size(img,3) == 1
                img = repmat(img, [1 1 3]);
            end

        case '.mat'
            S = load(path);
            if ~isfield(S, 'img')
                error('Input .mat file does not contain "img": %s', path);
            end
            img = double(S.img);
            if size(img,3) == 1
                img = repmat(img, [1 1 3]);
            end

        otherwise
            error('Unsupported input format: %s', ext);
    end
end

function [sorted, idx] = sort_nat(strings)
    if isempty(strings)
        sorted = strings;
        idx = [];
        return;
    end

    strings = string(strings(:));
    num_tokens = regexp(strings, '\d+', 'match');
    max_tokens = max(cellfun(@numel, num_tokens));
    num_mat = zeros(numel(strings), max_tokens);
    for i = 1:numel(strings)
        toks = num_tokens{i};
        for j = 1:numel(toks)
            num_mat(i, j) = str2double(toks{j});
        end
    end
    base_keys = lower(regexprep(strings, '\d+', ' '));
    sort_keys = base_keys;
    for j = 1:max_tokens
        sort_keys = sort_keys + " " + string(num_mat(:, j));
    end
    [~, idx] = sort(sort_keys);
    sorted = strings(idx);
end

function [y_norm, y_min, y_max] = nor_minmax(y)
    y_min = min(y(:));
    y_max = max(y(:));
    rangev = (y_max - y_min) + 1e-12;
    y_norm = (y - y_min) / rangev;
end

function out = srgb_gamma_correction(in)
    out = zeros(size(in));
    mask = in <= 0.0031308;
    out(mask) = 12.92 * in(mask);
    out(~mask) = 1.055 * (in(~mask) .^ (1 / 2.4)) - 0.055;
end

function comp_denominator_values = compute_comp_denominators_lnmax_times_1024(input_files)
    N = numel(input_files);
    comp_denominator_values = zeros(N, 1);

    for k = 1:N
        inputPath = fullfile(input_files(k).folder, input_files(k).name);
        img = read_input_rgb(inputPath);
        max_val = max(img(:));

        if max_val <= 0
            warning('Input max <= 0 for %s, fallback max to exp(1).', input_files(k).name);
            max_val = exp(1);
        end

        denom = log(max_val) * 1024;
        if abs(denom) < 1e-12
            warning('ln(max)*1024 is near zero for %s, fallback denominator to 1e-12.', input_files(k).name);
            denom = 1e-12;
        end

        if ~isfinite(denom)
            warning('Invalid denominator for %s, fallback denominator to 1e-12.', input_files(k).name);
            denom = 1e-12;
        end

        comp_denominator_values(k) = denom;
    end
end
