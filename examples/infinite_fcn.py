# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""An example doing inference with an infinitely wide fully-connected network.

By default, this example does inference on a small CIFAR10 subset.
"""

import time
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util


flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 1000,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 0,
                     'Batch size for kernel computation. 0 for no batching.')


FLAGS = flags.FLAGS


def main(unused_argv):
  # Build data pipelines.
  print('Loading data.')
  x_train, y_train, x_test, y_test = \
    datasets.get_dataset('cifar10', FLAGS.train_size, FLAGS.test_size)

  # Build the infinite network.
  _, _, kernel_fn = stax.serial(
      stax.Dense(1, 2., 0.05),
      stax.Relu(),
      stax.Dense(1, 2., 0.05)
  )

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn,
                       device_count=0,
                       batch_size=FLAGS.batch_size)

  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      x_test,
                                                      get=('nngp', 'ntk'),
                                                      diag_reg=1e-3)
  fx_test_nngp.block_until_ready()
  fx_test_ntk.block_until_ready()

  duration = time.time() - start
  print('Kernel construction and inference done in %s seconds.' % duration)

  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
  util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)


if __name__ == '__main__':
  app.run(main)
