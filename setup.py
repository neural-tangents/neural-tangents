# Copyright 2019 The Neural Tangents Authors.  All rights reserved.
import setuptools


INSTALL_REQUIRES = [
    'absl-py',
    'numpy',
    'jax',
    'aenum',
]

setuptools.setup(
    name='neural-tangents',
    version='0.0.0',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/neural-tangents/neural-tangents',
    packages=setuptools.find_packages()
    )
