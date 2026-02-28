#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImgStegGAN V1.0.0 Setup Script

This project is a modified and extended version of SteganoGAN
by MIT Data To AI Lab (https://github.com/DAI-Lab/SteganoGAN)

Original paper: Zhang, Kevin Alex and Cuesta-Infante, Alfredo and 
Veeramachaneni, Kalyan. SteganoGAN: High Capacity Image Steganography 
with GANs. MIT EECS, January 2019. (arXiv:1901.03892)
"""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

install_requires = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'Pillow>=9.0.0',
    'numpy>=1.21.0',
    'reedsolo>=1.0.0',
    'tqdm>=4.64.0',
    'flask>=2.0.0',
    'flask-cors>=3.0.0',
    'transformers>=4.30.0',
]

setup(
    author="ImgStegGAN Team",
    author_email='imgsteggan@example.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="GAN-Driven Image Steganography with Qwen Enhancement (based on SteganoGAN)",
    entry_points={
        'console_scripts': [
            'imgsteggan=steganogan.cli:main'
        ],
    },
    install_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='imgsteggan steganography deep-learning gan pytorch qwen steganogan',
    name='imgsteggan',
    packages=find_packages(include=['steganogan', 'steganogan.*']),
    python_requires='>=3.8',
    url='https://github.com/yourusername/ImgStegGAN',
    version='1.0.0',
    zip_safe=False,
)
