@echo off
setlocal

set "ENV_NAME=fumo_official"
set "PROJECT_DIR=D:\ResearchCode\halo-free\Decoupled_Learning\20260916_refremoval"
set "CONDA_EXE=D:\anaconda\Scripts\conda.exe"

echo Creating conda environment: %ENV_NAME%
%CONDA_EXE% env list | findstr /C:"%ENV_NAME%" >nul
if errorlevel 1 (
  %CONDA_EXE% create -y -n %ENV_NAME% python=3.10
  if errorlevel 1 exit /b %ERRORLEVEL%
) else (
  echo Environment %ENV_NAME% already exists; continuing installation.
)

echo Verifying Python version
%CONDA_EXE% run --no-capture-output -n %ENV_NAME% python -c "import sys; print(sys.version); assert sys.version_info[:2] == (3, 10), 'fumo_official must use Python 3.10'"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Installing PyTorch 2.4.1 + CUDA 12.1 wheels
%CONDA_EXE% run --no-capture-output -n %ENV_NAME% python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b %ERRORLEVEL%

echo Installing FUMO Windows-compatible dependencies
%CONDA_EXE% run --no-capture-output -n %ENV_NAME% python -m pip install ^
  accelerate==1.1.1 ^
  diffusers==0.32.0 ^
  transformers==4.46.2 ^
  huggingface-hub==0.36.0 ^
  safetensors==0.6.2 ^
  tokenizers==0.20.3 ^
  datasets==4.4.1 ^
  einops==0.8.2 ^
  lpips==0.1.4 ^
  lmdb==1.7.5 ^
  opencv-python==4.12.0.88 ^
  pillow==10.4.0 ^
  numpy==2.2.6 ^
  scipy==1.15.3 ^
  scikit-image==0.25.2 ^
  imageio==2.36.1 ^
  matplotlib==3.10.7 ^
  tensorboard==2.20.0 ^
  tqdm==4.67.1 ^
  pandas==2.2.3 ^
  PyYAML==6.0.3 ^
  requests==2.32.5 ^
  regex==2025.11.3 ^
  rich==14.2.0 ^
  psutil==5.9.8 ^
  protobuf==6.33.2 ^
  gradio==4.44.1 ^
  gradio_client==1.3.0 ^
  gradio_imageslider==0.0.15 ^
  fastapi==0.121.0 ^
  uvicorn==0.38.0 ^
  pydantic==2.12.4 ^
  aiohttp==3.13.2 ^
  pyarrow==22.0.0 ^
  multiprocess==0.70.18 ^
  xxhash==3.6.0 ^
  hf-xet==1.2.0 ^
  hf_transfer==0.1.9 ^
  modelscope==1.31.0 ^
  peft==0.17.1
if errorlevel 1 exit /b %ERRORLEVEL%

echo Verifying versions
%CONDA_EXE% run --no-capture-output -n %ENV_NAME% python -c "import torch, torchvision, accelerate, diffusers, transformers, lpips; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('torchvision', torchvision.__version__); print('accelerate', accelerate.__version__); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Done. Environment %ENV_NAME% is ready.
