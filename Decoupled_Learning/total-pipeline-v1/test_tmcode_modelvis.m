clear; clc;
% 合并版：
% 1) 按 test_tmcode_wlsvis 的逻辑输出 WLS 结果，并提取 minB_values
% 2) 按 test_tmcode_modelvis 的逻辑使用 modelOutDir 的 base 层做可视化

%% ====== Path Configuration ======
modelOutDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\rf3stage_v8_output';
inputDir = 'D:\nuoqi\S2R-HDR\data\test_vivo\hdrimg';
outputPath_wls_rgb = 'D:\nuoqi\halo-free\Decoupled_Learning\total-pipeline-v1\vis';
outputPath_model_rgb = 'D:\nuoqi\halo-free\Decoupled_Learning\total-pipeline-v1\vis';

ensure_dir(outputPath_wls_rgb);
ensure_dir(outputPath_model_rgb);

%% ====== Constants ======
epsilon = 1e-10;
comp_numerator = 9937;
% WLS 参数
lambda = 0.25;
alpha = 1.2;

%% ====== Collect and Sort Files ======
model_exts = {'*.tiff','*.png','*.exr','*.mat'};
model_files = [];
for i = 1:numel(model_exts)
    model_files = [model_files; dir(fullfile(modelOutDir, model_exts{i}))]; %#ok<AGROW>
end
if isempty(model_files)
    error('No model output files found in: %s', modelOutDir);
end
model_names = {model_files.name};
[~, idx] = sort_nat(model_names);
model_files = model_files(idx);

input_exts = {'*.hdr','*.exr','*.mat','*.tiff','*.png'};
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

N = min([numel(model_files), numel(input_files)]);
if numel(model_files) ~= numel(input_files)
    warning(['File count mismatch detected:\n' ...
             '  Model outputs: %d\n' ...
             '  Input files: %d\n' ...
             'Model-visualization will process only the first %d pairs.'], ...
            numel(model_files), numel(input_files), N);
end

%% ====== WLS VIS + Extract minB Values ======
minB_values = run_wlsvis_and_extract_minb(input_files, outputPath_wls_rgb, epsilon, comp_values, lambda, alpha);
disp('Extracted minB_values from WLS pipeline (all input files):');
disp(minB_values');

%% ====== Model VIS Main Processing Loop ======
for k = 1:N
    compression_factor = comp_values(min(k, numel(comp_values)));

    fileModel = model_files(k).name;
    fileInput = input_files(k).name;

    % ---- Read model output as filtered base (log-domain, normalized) ----
    inPathFiltered = fullfile(modelOutDir, fileModel);
    img_filtered = read_model_filtered(inPathFiltered);

    % ---- Read original input ----
    inputPath = fullfile(inputDir, fileInput);
    img = read_input_rgb(inputPath);

    if size(img, 3) ~= 3
        disp(['Warning: image ', fileInput, ' is not 3-channel, skipping']);
        continue;
    end

    % ---- Global max normalization (testv3 logic) ----
    max_val = max(img(:));
    if max_val > 0
        img = img / max_val;
    end

    % ---- Intensity + log domain ----
    intensity = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
    intensity = max(intensity, epsilon);
    log_intensity = log(intensity);

    % ---- Reconstruct base_log from normalized model output ----
    [~, y_min, y_max] = nor_minmax(log_intensity);
    rangev = (y_max - y_min) + 1e-12;
    base_log = y_min + img_filtered * rangev;
    
    % ----- 图像对齐 -------
    if ~isequal(size(base_log), size(log_intensity))
        base_log = imresize(base_log, size(log_intensity), 'bilinear');
    end
    
    % ---- Tone mapping (testv3 logic) ----
    detail_log = log_intensity - base_log;
    target_log = (base_log * compression_factor) + detail_log;

    % follow blf 中的一个操作步骤，通过取最小1 % 并用一个经验数来
    % 
    minB = minB_values(min(k, numel(minB_values)));
    target_log = target_log - (5.4 + minB * compression_factor);

    detailed_channel = exp(target_log);
    ratio = detailed_channel ./ (intensity + epsilon);
    img_out = img .* ratio;

    % ---- sRGB gamma + quantize ----
    img_out = min(max(img_out, 0), 1);
    img_out_srgb = srgb_gamma_correction(img_out);
    img_u8 = uint8(img_out_srgb * 255);

    % ---- Save outputs ----
    [~, ~, ext] = fileparts(fileInput);
    if ~ismember(lower(ext), {'.tiff', '.tif'})
        ext = '.tiff';
    end

    seqName = sprintf('%04d', k);
    outputFileName_outputjpg = fullfile(outputPath_model_rgb, [seqName, '-3stage.jpg']);

    imwrite(img_u8, outputFileName_outputjpg);
end

disp('All WLS and model visualization images processed and saved');

%% ====== Helper Functions ======
function ensure_dir(path)
    if ~exist(path, 'dir')
        mkdir(path);
    end
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

    %img_filtered = min(max(img_filtered, 0), 1);
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

function write_tiff_rgb(file_path, data)
    t = Tiff(file_path, 'w');
    tagstruct = make_tiff_tags(size(data, 1), size(data, 2), 3, Tiff.Photometric.RGB);
    t.setTag(tagstruct);
    t.write(single(data));
    t.close();
end

function tagstruct = make_tiff_tags(height, width, samples_per_pixel, photometric)
    tagstruct = struct();
    tagstruct.ImageLength = height;
    tagstruct.ImageWidth = width;
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = photometric;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = samples_per_pixel;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;
end

% 标准 sRGB Gamma 曲线 (对齐 testv3_tmcode_copy)
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
    range = (y_max - y_min) + 1e-12;
    y_norm = (y - y_min) / range;
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

        % 与 wlsvis 对齐：全局 max 归一化 -> intensity -> log -> minmax -> WLS -> tone mapping
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

        fprintf('WLS Processed: %s\n', name);
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
