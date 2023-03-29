import platform
from setuptools import setup, find_packages


install_requires = [
    "lightning==1.9.0",
    "accelerate==0.18.0",
    "bitsandbytes==0.37.0",
    "bitsandbytes-cuda117==0.26.0.post2",
    "controlnet_aux",
    "huggingface-hub==0.13.3",
    "numpy==1.23.5",
    "Pillow==9.4.0",
    "pip==23.0.1",
    "PyQt6==6.4.2",
    "PyQt6-Qt6==6.4.3",
    "PyQt6-sip==13.4.1",
    "pyqtdarktheme==2.1.0",
    "pyre-extensions==0.0.23",
    "pytorch-lightning==2.0.0",
    "requests==2.28.2",
    "requests-oauthlib==1.3.1",
    "safetensors==0.3.0",
    "scipy==1.10.1",
    "tensorflow==2.12.0",
    "tokenizers==0.13.2",
    "tqdm==4.65.0",
    "xformers==0.0.16",
    "charset-normalizer==3.1.0",
    "opencv-python==4.7.0.72",
    "setuptools==65.5.1",
    "sympy==1.11.1",
    "typing_extensions==4.5.0",
    "urllib3==1.26.15",
]

if platform.system() == "Linux":
    install_requires = [
        "torch==1.13.1",
        "torchvision==0.14.1",
        "torchaudio==0.13.1",
        "nvidia-pyindex",
        "nvidia-tensorrt==8.4.3.1",
        "triton==2.0.0.dev20221202",
        "deepspeed==0.8.0",
    ] + install_requires

setup(
    name="aihandler",
    version="1.8.14",
    author="Capsize LLC",
    description="AI Handler: An engine which wraps certain huggingface models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="ai, chatbot, chat, ai",
    license="AGPL-3.0",
    author_email="contact@capsize.gg",
    url="https://github.com/w4ffl35/aihandler",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=install_requires,
    extras_require={},
)