# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""Describes flags used by neural tangents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags


flags.DEFINE_boolean(
    'tangents_optimized', True,
    '''Flag sets whether or not (potentially unsafe) optimizations are used.'''
)
