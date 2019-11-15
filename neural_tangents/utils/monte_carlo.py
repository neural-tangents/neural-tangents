# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""Methods to compute Monte Carlo NNGP and NTK estimates.

The library has a public method `monte_carlo_kernel_fn` that allow to compute
  Monte Carlo estimates of NNGP and NTK kernels of arbitrary functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax import random
from functools import partial
import operator
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from neural_tangents.utils import batch
from neural_tangents.utils import empirical
from neural_tangents.utils.utils import get_namedtuple


def _sample_once_kernel_fn(kernel_fn,
                           init_fn,
                           batch_size=0,
                           device_count=-1,
                           store_on_device=True):
  @partial(batch.batch, batch_size=batch_size,
           device_count=device_count,
           store_on_device=store_on_device)
  def kernel_fn_sample_once(x1, x2, key, get):
    _, params = init_fn(key, x1.shape)
    return kernel_fn(x1, x2, params, get)
  return kernel_fn_sample_once


def _sample_many_kernel_fn(kernel_fn_sample_once, key, n_samples,
                           get_generator):
  def normalize(sample, n):
    return tree_map(lambda sample: sample / n, sample)

  def get_samples(x1, x2, get):
    if x2 is not None:
      assert x1.shape[1:] == x2.shape[1:]

    _key = key
    for n in range(1, max(n_samples) + 1):
      _key, split = random.split(_key)
      one_sample = kernel_fn_sample_once(x1, x2, split, get)
      if n == 1:
        ker_sampled = one_sample
      else:
        ker_sampled = tree_multimap(operator.add, ker_sampled, one_sample)
      yield n, ker_sampled

  if get_generator:
    @get_namedtuple('MonteCarloKernel')
    def get_sampled_kernel(x1, x2, get=None):
      for n, sample in get_samples(x1, x2, get):
        if n in n_samples:
          yield normalize(sample, n)
  else:
    @get_namedtuple('MonteCarloKernel')
    def get_sampled_kernel(x1, x2, get=None):
      for n, sample in get_samples(x1, x2, get):
        pass
      return normalize(sample, n)

  return get_sampled_kernel


def monte_carlo_kernel_fn(init_fn,
                          apply_fn,
                          key,
                          n_samples,
                          batch_size=0,
                          device_count=-1,
                          store_on_device=True):
  """Return a Monte Carlo sampler of NTK and NNGP kernels of a given function.

  Args:
    init_fn: a function initializing parameters of the neural network. From
      `jax.experimental.stax`: "takes an rng key and an input shape and returns
      an `(output_shape, params)` pair".
    apply_fn: a function computing the output of the neural network.
      From `jax.experimental.stax`: "takes params, inputs, and an rng key and
      applies the layer".
    key: RNG (`jax.random.PRNGKey`) for sampling random networks. Must have
      shape `(2,)`.
    n_samples: number of Monte Carlo samples. Can be either an integer or an
      iterable of integers at which the resulting generator will yield
      estimates. Example: use `n_samples=[2**k for k in range(10)]` for the
      generator to yield estimates using 1, 2, 4, ..., 512 Monte Carlo samples.
    batch_size: an integer making the kernel computed in batches of `x1` and
      `x2` of this size. `0` means computing the whole kernel. Must divide
      `x1.shape[0]` and `x2.shape[0]`.
    device_count: an integer making the kernel be computed in parallel across
      this number of devices (e.g. GPUs or TPU cores). `-1` means use all
      available devices. `0` means compute on a single device sequentially. If
      not `0`, must divide `x1.shape[0]`.
    store_on_device: a boolean, indicating whether to store the resulting
      kernel on the device (e.g. GPU or TPU), or in the CPU RAM, where larger
      kernels may fit.

  Returns:
    If `n_samples` is an integer, returns a function of signature
    `kernel_fn(x1, x2, get)` that returns an MC estimation of the kernel using
    `n_samples`. If `n_samples` is a collection of integers,
    `kernel_fn(x1, x2, get)` returns a generator that yields estimates using
    `n` samples for `n in n_samples`.

  Example:
  ```python
  >>> from jax import random
  >>> import neural_tangents as nt
  >>> from neural_tangents import stax
  >>>
  >>> key1, key2 = random.split(random.PRNGKey(1), 2)
  >>> x_train = random.normal(key1, (20, 32, 32, 3))
  >>> y_train = random.uniform(key1, (20, 10))
  >>> x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>> init_fn, apply_fn, kernel_fn = stax.serial(
  >>>     stax.Conv(128, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(256, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(512, (3, 3)),
  >>>     stax.Flatten(),
  >>>     stax.Dense(10)
  >>> )
  >>>
  >>> n_samples = 200
  >>> kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key1, n_samples)
  >>> kernel = kernel_fn(x_train, x_test, get=('nngp', 'ntk'))
  >>> # `kernel` is a tuple of NNGP and NTK MC estimate using `n_samples`.
  >>>
  >>> n_samples = [1, 10, 100, 1000]
  >>> kernel_fn_generator = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key1,
  >>>                                                n_samples)
  >>> kernel_samples = kernel_fn_generator(x_train, x_test, get=('nngp', 'ntk'))
  >>> for n, kernel in zip(n_samples, kernel_samples):
  >>>   print(n, kernel)
  >>>   # `kernel` is a tuple of NNGP and NTK MC estimate using `n` samples.
  ```
  """
  kernel_fn = empirical.empirical_kernel_fn(apply_fn)

  kernel_fn_sample_once = _sample_once_kernel_fn(kernel_fn,
                                                 init_fn,
                                                 batch_size,
                                                 device_count,
                                                 store_on_device)

  n_samples, get_generator = _canonicalize_n_samples(n_samples)
  kernel_fn = _sample_many_kernel_fn(kernel_fn_sample_once, key, n_samples,
                                     get_generator)
  return kernel_fn


def _canonicalize_n_samples(n_samples):
  get_generator = True
  if isinstance(n_samples, int):
    get_generator = False
    n_samples = (n_samples,)

  if hasattr(n_samples, '__iter__'):
    n_samples = set(n_samples)

    if not all(isinstance(n, int) for n in n_samples):
      raise ValueError('`n_samples` must contain only integers, got %s.'
                       % n_samples)

    if any(n <= 0 for n in n_samples):
      raise ValueError('`n_samples` must be positive, got %s.' % n_samples)

  else:
    raise ValueError('`n_samples` must be either an integer of a set of '
                     'integers, got %s.' % type(n_samples))
  return n_samples, get_generator
