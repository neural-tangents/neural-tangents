# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

import io
import os
import setuptools

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


INSTALL_REQUIRES = [
    'jaxlib>=0.1.32',
    'jax>=0.1.50',
]


setuptools.setup(
    name='neural-tangents',
    version='0.0.0',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/neural-tangents/neural-tangents',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Fast and Easy Infinite Neural Networks in Python',
    python_requires='>=2.7',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ]
)
