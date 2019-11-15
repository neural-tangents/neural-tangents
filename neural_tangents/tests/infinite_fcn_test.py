# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""Tests for `examples/function_space.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax import test_util as jtu
from jax.config import config
from examples import infinite_fcn


config.parse_flags_with_absl()


class InfiniteFcnTest(jtu.JaxTestCase):

  def test_infinite_fcn(self):
    infinite_fcn.main(None)


if __name__ == '__main__':
  jtu.absltest.main()
