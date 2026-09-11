inputPath =  'D:\nuoqi\S2R-HDR\data\alldata_origin\HDR_gt_sihdr';  % 输入 HDR 图像路径
baseOutputPath = 'D:\nuoqi\S2R-HDR\data\20260813';
outputPath = fullfile(baseOutputPath, 'testlabel-l1p000-a1p600');
logOutputPath = fullfile(baseOutputPath, 'testinput-l1p000-a1p600');
%outputPath = 'Y:/nuoqiwang/8.5/HDR_trainlabel_log';
files = dir(fullfile(inputPath, '*.hdr'));
if isempty(files)
    files = dir(fullfile(inputPath, '*.HDR'));
end
if isempty(files)
    error('No HDR files found in inputPath: %s', inputPath);
end
%imageFiles = dir(fullfile(inputPath, '*.jpg'));
%parameterStr = '021544';g
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
if ~exist(logOutputPath, 'dir')
    mkdir(logOutputPath);
end

% 每张图的读取、WLS 和写入完全独立，因此可安全并行。
% 若已有并行池则复用；没有时创建最多 8 个 local worker。
numWorkers = 16;
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local', numWorkers);
end

parfor k = 1:length(files)
    %RGB = imread(inputImagePath);    % 读取图像
    inputImagePath = fullfile(inputPath, files(k).name);
    [~, ~, ext] = fileparts(inputImagePath);
    switch lower(ext)
        case '.hdr'
            img = double(hdrread(inputImagePath));
        otherwise
            error('Unsupported input extension: %s', ext);
    end
    max_val = max(img(:));
    if max_val > 0
        img = img / max_val;
    end
    % R = img(:,:,1);
    % G = img(:,:,2);
    % B = img(:,:,3);

    img = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
    img = double(img);
    % 2. 设置参数
    lambda = 1.0;  % 平滑参数
    alpha = 1.6;   % 边缘保留参数

    % 3. 调用 wlsFilter 函数
    %res = wls(I, lambda, alpha); %灰度
    %img = log(img + 1e-6);
%     img = nor(img);
%     img = img * 255;
    %img = log(img + 1e-6);
    eps_val = 1e-10;
    linear_img = max(img, eps_val);
    log_img = 1024 * log(linear_img);
    log_img_min = min(log_img(:));
    log_img_max = max(log_img(:));
    log_img_range = log_img_max - log_img_min;
    if log_img_range < 1e-12
        log_img = zeros(size(log_img));
    else
        log_img = (log_img - log_img_min) / log_img_range;
    end
    img_filtered = wls(log_img, lambda, alpha);
    %img_filtered = nor(img_filtered);
%     img_filtered(:,:,2) = wls(img(:,:,2),lambda, alpha);
%     img_filtered(:,:,3) = wls(img(:,:,3),lambda, alpha);
    %img_filtered = nor(img_filtered);
    [~, baseName, ~] = fileparts(files(k).name);
    ext = '.tiff';
    % 去除 "-input" 部分
    %baseName = strrep(baseName, '-input', '');
    % 构造新的文件名，例如 "xxx-wls-010000.png"
    outputFileName = fullfile(outputPath, [baseName, '-WLS-l1p000-a1p600', ext]);
    logOutputFileName = fullfile(logOutputPath, [baseName, '-log', ext]);

    % 保存滤波后的图像
    %imwrite(img_filtered, outputFileName);
    t_l = Tiff(outputFileName, 'w');
    tagstruct = struct();
    tagstruct.ImageLength = size(img_filtered, 1);
    tagstruct.ImageWidth = size(img_filtered, 2);
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Compression = Tiff.Compression.None;
    
    t_l.setTag(tagstruct);
    t_l.write(single(img_filtered));
    t_l.close();

    % 保存 log_img
    t_log = Tiff(logOutputFileName, 'w');
    tagstruct_log = struct();
    tagstruct_log.ImageLength = size(log_img, 1);
    tagstruct_log.ImageWidth = size(log_img, 2);
    tagstruct_log.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct_log.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct_log.BitsPerSample = 32;
    tagstruct_log.SamplesPerPixel = 1;
    tagstruct_log.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct_log.Compression = Tiff.Compression.None;
    t_log.setTag(tagstruct_log);
    t_log.write(single(log_img));
    t_log.close();
end
disp('所有图像已处理并保存！');

% % 4. 显示原始图像和滤波结果
% figure;
% subplot(1, 3, 1);
% imshow(img);
% title('原始图像');
% 
% subplot(1, 3, 2);
% imshow(wls_filtered_RGB);
% title('WLS滤波结果');
% 
% subplot(1, 3, 3);
% imshow(blf_filtered_RGB);
% title('BLF滤波结果');
