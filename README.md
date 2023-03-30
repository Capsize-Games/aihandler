# AI Handler
[![Upload Python Package](https://github.com/Capsize-Games/aihandler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Capsize-Games/aihandler/actions/workflows/python-publish.yml)
[![Discord](https://img.shields.io/discord/839511291466219541?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/PUVDDCJ7gz)
![GitHub](https://img.shields.io/github/license/Capsize-Games/aihandler)
![GitHub last commit](https://img.shields.io/github/last-commit/Capsize-Games/aihandler)
![GitHub issues](https://img.shields.io/github/issues/Capsize-Games/aihandler)
![GitHub closed issues](https://img.shields.io/github/issues-closed/Capsize-Games/aihandler)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Capsize-Games/aihandler)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Capsize-Games/aihandler)

This is a simple framework for running AI models. It makes use of the huggingface API
which gives you a queue, threading, a simple API, and the ability to run Stable Diffusion and LLMs seamlessly
from your local hardware.

This is not intended to be used as a standalone application.

It can easily be extended and used to power interfaces or it can be run from the command line.

AI Handler is a work in progress. It powers two projects at the moment, but may not be ready for general use.

## Installation

This is a work in progress.

## Pre-requisites

System requirements

- Python 3.10.8
- pip 23.0.1
- CUDA toolkit 11.7
- CUDNN 8.6.0.163
- Cuda capable GPU
- 16gb+ ram

### Ubuntu 20.04+

Install required libraries
```
pip install https://github.com/w4ffl35/diffusers/archive/refs/tags/v0.14.0.ckpt_fix.tar.gz
pip install https://github.com/w4ffl35/transformers/archive/refs/tags/tensor_fix-v1.0.2.tar.gz
pip install aihandler
```

### Windows 10+

```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
pip install https://github.com/w4ffl35/diffusers/archive/refs/tags/v0.14.0.ckpt_fix.tar.gz
pip install https://github.com/w4ffl35/transformers/archive/refs/tags/tensor_fix-v1.0.2.tar.gz
pip install https://github.com/acpopescu/bitsandbytes/releases/download/v0.37.2-win.0/bitsandbytes-0.37.2-py3-none-any.whl
pip install aihandler
```

---

Currently bitsandbytes on windows is bitsandbroken. Here's how you can hack around it:

1. `git clone https://github.com/DeXtmL/bitsandbytes-win-prebuilt`
2. `git clone https://github.com/james-things/bitsandbytes-prebuilt-all_arch`
3. `copy bitsandbytes-win-prebuilt/*.dll <your_venv_path>/site-packages/bitsandbytes`
4. `copy bitsandbytes-prebuilt-all_arch/*.dll <your_venv_path>/site-packages/bitsandbytes`
5. Edit `main.py` in `<your_venv_path>/site-packages/bitsandbytes/cuda_setup` 
6. Find and replace all `ct.cdll.LoadLibrary(binary_path)` with `ct.cdll.LoadLibrary(str(binary_path))`
7. Find and replace all `if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.so', None, None, None, None` with `if torch.cuda.is_available(): return 'libbitsandbytes_cudaall.dll', None, None, None, None`

#### Optional

These are optional instructions for installing TensorRT and Deepspeed for Windows

##### Install Tensor RT:

1. Download TensorRT-8.4.3.1.Windows10.x86_64.cuda-11.6.cudnn8.4
2. Git clone TensorRT 8.4.3.1
3. Follow their instructions to build TensorRT-8.4.3.1 python wheel
4. Install TensorRT `pip install tensorrt-*.whl`
 
##### Install Deepspeed:

1. Git clone Deepspeed 0.8.1
2. Follow their instructions to build Deepspeed python wheel
3. Install Deepspeed `pip install deepspeed-*.whl

---

## Environment variables

- `AIRUNNER_ENVIRONMENT` - `dev` or `prod`. Defaults to `dev`. This controls the LOG_LEVEL
- `LOG_LEVEL` - `FATAL` for production, `DEBUG` for development. Override this to force a log level

### Huggingface variables

#### Offline mode

These environment variables keep you offline until you need to download a model. This prevents unwanted online access and speeds up usage of huggingface libraries.

- `DISABLE_TELEMETRY` Keep this set to 1 at all times. Huggingface collects minimal telemetry when downloading a model from their repository but this will keep it disabled. [See more info in this github thread](https://github.com/huggingface/diffusers/pull/1833#issuecomment-1368484414)
- `HF_HUB_OFFLINE` When loading a diffusers model, huggingface libraries will attempt to download an updated cache before running the model. This prevents that check from happening (long with a boolean passed to `load_pretrained` see the runner.py file for examples)
- `TRANSFORMERS_OFFLINE` Similar to `HF_HUB_OFFLINE` but for transformers models
