# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""Tests for `examples/weight_space.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax import test_util as jtu
from jax.config import config
from examples import weight_space


config.parse_flags_with_absl()


class WeightSpaceTest(jtu.JaxTestCase):

  def test_weight_space(self):
    weight_space.main(None)


if __name__ == '__main__':
  jtu.absltest.main()
