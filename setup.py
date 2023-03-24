"""
A standard python setup.py file for aiengine that allows users to install with pip
"""
from setuptools import setup, find_packages

extras = {}
install_requires = [
]

setup(
    name='aiengine',
    version='1.8.9',
    author='Capsize LLC',
    description="AI Engine",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="",
    keywords="",
    license="",
    author_email="contact@capsize.gg",
    url="https://github.com/w4ffl35/aiengine",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=install_requires,
    extras_require=extras,
)
