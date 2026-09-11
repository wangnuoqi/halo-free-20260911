clear; clc;
% 多模型目录可视化：
% 1) 对 vivo HDR 输入先计算一次 WLS 基准和 minB
% 2) 遍历多个模型输出目录
% 3) 每个模型目录单独输出自己的 3 张可视化结果

%% ====== Path Configuration ======
modelOutRoot = 'D:\nuoqi\halo-free\Decoupled_Learning\20260705_vis\vis_comp';
inputDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
outputRoot = 'D:\nuoqi\halo-free\Decoupled_Learning\total-pipeline-v1\vis_multi_models';
outputPath_wls_rgb = fullfile(outputRoot, 'wls_gt');

ensure_dir(outputRoot);
ensure_dir(outputPath_wls_rgb);

modelOutDirs = collect_model_output_dirs(modelOutRoot);
if isempty(modelOutDirs)
    error('No model output subdirectories found in: %s', modelOutRoot);
end

%% ====== Constants ======
epsilon = 1e-10;
comp_numerator = 9937;
lambda = 0.25;
alpha = 1.2;

%% ====== Collect and Sort Input Files ======
input_exts = {'*.hdr','*.exr','*.mat','*.tiff','*.tif','*.png'};
input_files = [];
for i = 1:numel(input_exts)
    input_files = [input_files; dir(fullfile(inputDir, input_exts{i}))]; %#ok<AGROW>
end
if isempty(input_files)
    error('No input files found in: %s', inputDir);
end

input_names = {input_files.name};
[~, idx] = sort_nat(input_names);
input_files = input_files(idx);

comp_denominator_values = compute_comp_denominators_lnmax_times_1024(input_files);
comp_values = comp_numerator ./ comp_denominator_values;
disp('Auto-computed comp_values = 9937 ./ (ln(max(input)) * 1024):');
disp(comp_values');

%% ====== WLS VIS + Extract minB Values ======
minB_values = run_wlsvis_and_extract_minb(input_files, outputPath_wls_rgb, epsilon, comp_values, lambda, alpha);
disp('Extracted minB_values from WLS pipeline (all input files):');
disp(minB_values');

%% ====== Process Each Model Directory ======
for dir_idx = 1:numel(modelOutDirs)
    modelOutDir = modelOutDirs{dir_idx};
    model_tag = sanitize_name(get_last_path_part(modelOutDir));
    outputPath_model_rgb = fullfile(outputRoot, model_tag);
    ensure_dir(outputPath_model_rgb);

    fprintf('\n===== Processing model directory: %s =====\n', modelOutDir);

    model_exts = {'*.tiff','*.tif','*.png','*.exr','*.mat'};
    model_files = [];
    for i = 1:numel(model_exts)
        model_files = [model_files; dir(fullfile(modelOutDir, model_exts{i}))]; %#ok<AGROW>
    end
    if isempty(model_files)
        warning('No model output files found in: %s. Skipping.', modelOutDir);
        continue;
    end

    model_names = {model_files.name};
    [~, idx] = sort_nat(model_names);
    model_files = model_files(idx);

    N = min([numel(model_files), numel(input_files)]);
    if numel(model_files) ~= numel(input_files)
        warning(['File count mismatch detected for model directory:\n' ...
                 '  Model outputs: %d\n' ...
                 '  Input files: %d\n' ...
                 'Only the first %d pairs will be processed for %s.'], ...
                numel(model_files), numel(input_files), N, modelOutDir);
    end

    for k = 1:N
        compression_factor = comp_values(min(k, numel(comp_values)));

        fileModel = model_files(k).name;
        fileInput = input_files(k).name;

        inPathFiltered = fullfile(modelOutDir, fileModel);
        img_filtered = read_model_filtered(inPathFiltered);

        inputPath = fullfile(input_files(k).folder, fileInput);
        img = read_input_rgb(inputPath);

        if size(img, 3) ~= 3
            warning('Image %s is not 3-channel, skipping.', fileInput);
            continue;
        end

        max_val = max(img(:));
        if max_val > 0
            img = img / max_val;
        end

        intensity = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
        intensity = max(intensity, epsilon);
        log_intensity = log(intensity);

        [~, y_min, y_max] = nor_minmax(log_intensity);
        rangev = (y_max - y_min) + 1e-12;
        base_log = y_min + img_filtered * rangev;

        if ~isequal(size(base_log), size(log_intensity))
            base_log = imresize(base_log, size(log_intensity), 'bilinear');
        end

        detail_log = log_intensity - base_log;
        target_log = (base_log * compression_factor) + detail_log;

        minB = minB_values(min(k, numel(minB_values)));
        target_log = target_log - (5.4 + minB * compression_factor);

        detailed_channel = exp(target_log);
        ratio = detailed_channel ./ (intensity + epsilon);
        img_out = img .* ratio;

        img_out = min(max(img_out, 0), 1);
        img_out_srgb = srgb_gamma_correction(img_out);
        img_u8 = uint8(img_out_srgb * 255);

        [~, input_name, ~] = fileparts(fileInput);
        outputFileName = fullfile(outputPath_model_rgb, sprintf('%02d_%s.jpg', k, input_name));
        imwrite(img_u8, outputFileName);

        fprintf('Saved %s -> %s\n', fileModel, outputFileName);
    end
end

disp('All multi-directory model visualization images processed and saved.');

%% ====== Helper Functions ======
function ensure_dir(pathstr)
    if ~exist(pathstr, 'dir')
        mkdir(pathstr);
    end
end

function model_dirs = collect_model_output_dirs(root_dir)
    entries = dir(root_dir);
    is_valid = [entries.isdir] & ~ismember({entries.name}, {'.', '..'});
    entries = entries(is_valid);

    model_dirs = {};
    for i = 1:numel(entries)
        subdir = fullfile(entries(i).folder, entries(i).name);
        has_tiff = ~isempty(dir(fullfile(subdir, '*.tiff'))) || ~isempty(dir(fullfile(subdir, '*.tif')));
        if has_tiff
            model_dirs{end + 1, 1} = subdir; %#ok<AGROW>
        end
    end
end

function part = get_last_path_part(pathstr)
    normalized = regexprep(pathstr, '[\\/]+$', '');
    tokens = regexp(normalized, '[\\/]', 'split');
    part = tokens{end};
end

function safe = sanitize_name(name)
    safe = regexprep(name, '[^\w\-]+', '_');
end

function img_filtered = read_model_filtered(pathstr)
    [~,~,ext] = fileparts(pathstr);
    ext = lower(ext);

    switch ext
        case {'.tif','.tiff','.png'}
            x = imread(pathstr);
            x = im2single(x);
            if ndims(x) == 3
                x = rgb2gray(x);
            end
            img_filtered = x;

        case '.exr'
            x = exrread(pathstr);
            x = single(x);
            if ndims(x) == 3
                x = rgb2gray(x);
            end
            img_filtered = x;

        case '.mat'
            S = load(pathstr);
            cand = {'img_filtered','pred','output','y','base'};
            found = '';
            for i = 1:numel(cand)
                if isfield(S, cand{i})
                    found = cand{i};
                    break;
                end
            end
            if isempty(found)
                error('Model output .mat file does not contain expected variables: %s', pathstr);
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

function img = read_input_rgb(pathstr)
    [~,~,ext] = fileparts(pathstr);
    ext = lower(ext);

    switch ext
        case '.hdr'
            img = hdrread(pathstr);
            img = double(img);

        case '.exr'
            img = exrread(pathstr);
            img = double(img);

        case {'.tif','.tiff','.png'}
            img = double(imread(pathstr));
            if size(img,3) == 1
                img = repmat(img, [1 1 3]);
            end

        case '.mat'
            S = load(pathstr);
            if ~isfield(S, 'img')
                error('Input .mat file does not contain "img": %s', pathstr);
            end
            img = double(S.img);
            if size(img,3) == 1
                img = repmat(img, [1 1 3]);
            end

        otherwise
            error('Unsupported input format: %s', ext);
    end
end

function out = srgb_gamma_correction(in)
    out = zeros(size(in));
    mask = in <= 0.0031308;
    out(mask) = 12.92 * in(mask);
    out(~mask) = 1.055 * (in(~mask) .^ (1 / 2.4)) - 0.055;
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

function minB_values = run_wlsvis_and_extract_minb(input_files, outputPath_wls_rgb, epsilon, comp_values, lambda, alpha)
    N = numel(input_files);
    minB_values = zeros(N, 1);

    for k = 1:N
        inputPath = fullfile(input_files(k).folder, input_files(k).name);
        img = read_input_rgb(inputPath);

        if size(img, 3) ~= 3
            error('Input image is not 3-channel for minB extraction: %s', input_files(k).name);
        end

        compression_factor = comp_values(min(k, numel(comp_values)));

        max_val = max(img(:));
        if max_val > 0
            img = img / max_val;
        end

        intensity = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
        intensity = max(intensity, epsilon);
        log_intensity = log(intensity);

        [log_intensity_nor, y_min, y_max] = nor_minmax(log_intensity);
        rangev = (y_max - y_min) + 1e-12;

        base_log_nor = wls(log_intensity_nor, lambda, alpha);
        base_log = y_min + base_log_nor * rangev;

        minB_values(k) = prctile(base_log(:), 1);

        detail_log = log_intensity - base_log;
        target_log = (base_log * compression_factor) + detail_log;
        target_log = target_log - (5.4 + minB_values(k) * compression_factor);

        detailed_channel = exp(target_log);
        ratio = detailed_channel ./ (intensity + epsilon);
        img_out = img .* ratio;

        img_out = min(max(img_out, 0), 1);
        img_out_srgb = srgb_gamma_correction(img_out);
        img_u8 = uint8(img_out_srgb * 255);

        [~, name, ~] = fileparts(input_files(k).name);
        outputFileName = fullfile(outputPath_wls_rgb, [name, '_gt.jpg']);
        imwrite(img_u8, outputFileName);
    end
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
