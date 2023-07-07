from setuptools import setup, find_packages


install_requires = [
    "einops==0.6.0",
    "ninja==1.11.1",
    "JIT==0.2.7",
    "tqdm==4.65.0",
    "omegaconf==2.3.0",
    "accelerate==0.20.3",
    "controlnet_aux==0.0.6",
    "huggingface-hub==0.16.3",
    "numpy==1.23.5",
    "Pillow==9.5.0",
    "pip==23.1.2",
    "PyQt6==6.4.2",
    "PyQt6-Qt6==6.4.3",
    "PyQt6-sip==13.4.1",
    "pyqtdarktheme==2.1.0",
    "pyre-extensions==0.0.29",
    "lightning==2.0.2",
    "requests==2.31.0",
    "requests-oauthlib==1.3.1",
    "safetensors==0.3.1",
    "scipy==1.10.1",
    "tensorflow==2.12.0",
    "tokenizers==0.13.3",
    "tqdm==4.65.0",
    "charset-normalizer==3.1.0",
    "opencv-python==4.8.0.74",
    "setuptools==67.7.2",
    "typing_extensions==4.5.0",
    "urllib3==1.26.15",
    "diffusers==0.18.0",
    "transformers==4.30.1",
    "compel==1.2.1",
    "sympy==1.12.0",
    "regex",
    "matplotlib==3.7.2",
]


setup(
    name="aihandler",
    version="1.17.4",
    author="Capsize LLC",
    description="AI Handler: An engine which wraps certain huggingface models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="ai, chatbot, chat, ai",
    license="AGPL-3.0",
    author_email="contact@capsize.gg",
    url="https://github.com/Capsize-Games/aihandler",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={},
)
