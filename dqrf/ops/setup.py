# Modified by Matthieu Lin
# Contact linmatthieu@gmail.com
# modified from https://github.com/idiap/fast-transformers/
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
import os
import glob
import torch
from setuptools import setup, Extension
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension

def get_extensions():
    extensions = [
        CppExtension(
            "DenseQuerySelfAttention",
            sources=[
                "./src/cpu/local_product_cpu.cpp"
            ],
            extra_compile_args=["-fopenmp", "-ffast-math"]
        ),

    ]
    return extensions


setup(
    name="DenseQuerySelfAttention",
    version="1.0",
    author="Matthieu Lin",
    url="https://github.com/Hatmm/PED-DETR-for-Pedestrian-Detection",
    description="Pytorch Wrapper for Functions of Dense Query",
    packages=find_packages("functions"),
    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
    install_requires=["torch"]
)