# AI Handler
[![Upload Python Package](https://github.com/Capsize-Games/aihandler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Capsize-Games/aihandler/actions/workflows/python-publish.yml)
![GitHub](https://img.shields.io/github/license/Capsize-Games/aihandler)
![GitHub last commit](https://img.shields.io/github/last-commit/Capsize-Games/aihandler)
![GitHub issues](https://img.shields.io/github/issues/Capsize-Games/aihandler)
![GitHub closed issues](https://img.shields.io/github/issues-closed/Capsize-Games/aihandler)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Capsize-Games/aihandler)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Capsize-Games/aihandler)

This is a simple library which can be used to run AI models. It is a light wrapper around the huggingface API
which gives you a queue, threading, a simple API, and the ability to run Stable Diffusion and LLMs seamlessly
from your local hardware.

It can easily be extended and used to power interfaces or it can be run from the command line.

AI Handler is a work in progress. It powers two projects at the moment, but may not be ready for general use.

## Installation

## Pre-requisites

System requirements

- Python 3.10.8
- pip 23.0.1
- CUDA toolkit 11.7
- CUDNN 8.6.0.163
- Cuda capable GPU
- 16gb+ ram

Create a venv and activate it:

**Ubuntu**

```
python -m venv venv
source venv/bin/activate
```

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

Upgrade pip to latest

`python.exe -m pip install --upgrade pip`

### Diffusers and transformers

Install modified versions of diffusers and transformers from these forks:

- pip install git+https://github.com/w4ffl35/diffusers.git@ckpt_fix
- pip install git+https://github.com/w4ffl35/transformers.git@tensor_fix

The author of `aihandler` is also the author of those changes.

### Ubuntu 20.04+

```
pip install aihandler
```

### Windows 10+

First install torch

```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```

Now install the rest

Install this repo `pip install aihandler`

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
