# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""General-purpose internal utilities."""

from jax.api import jit
from jax.api import vmap
from jax.lib import xla_bridge
import jax.numpy as np
from collections import namedtuple
from functools import wraps
import inspect
import types


def _jit_vmap(f):
  return jit(vmap(f))


def stub_out_pmap(batch, count):
  # If we are using GPU or CPU stub out pmap with vmap to simulate multi-core.
  if count > 0:
    class xla_bridge_stub(object):
      def device_count(self):
        return count

    platform = xla_bridge.get_backend().platform
    if platform == 'gpu' or platform == 'cpu':
      batch.pmap = _jit_vmap
      batch.xla_bridge = xla_bridge_stub()


def assert_close_matrices(self, expected, actual, rtol):
  self.assertEqual(expected.shape, actual.shape)
  relative_error = (np.linalg.norm(actual - expected) /
                    np.maximum(np.linalg.norm(expected), 1e-12))
  if relative_error > rtol or np.isnan(relative_error):
    self.fail(self.failureException(float(relative_error), expected, actual))
  else:
    print('PASSED with %f relative error.' % relative_error)


def canonicalize_get(get):
  if not get:
    raise ValueError('"get" must be non-empty.')

  get_is_not_tuple = isinstance(get, str)
  if get_is_not_tuple:
    get = (get,)

  get = tuple(s.lower() for s in get)
  if len(set(get)) < len(get):
    raise ValueError(
        'All entries in "get" must be unique. Got {}'.format(get))
  return get_is_not_tuple, get


_KERNEL_NAMED_TUPLE_CACHE = {}
def named_tuple_factory(name, get):
  key = (name, get)
  if key in _KERNEL_NAMED_TUPLE_CACHE:
    return _KERNEL_NAMED_TUPLE_CACHE[key]
  else:
    _KERNEL_NAMED_TUPLE_CACHE[key] = namedtuple(name, get)
    return named_tuple_factory(name, get)


def get_namedtuple(name):
  def getter_decorator(fn):
    try:
      get_index = inspect.getargspec(fn).args.index('get')
      defaults = inspect.getargspec(fn).defaults
    except:
      raise ValueError(
          '`get_namedtuple` functions must have a `get` argument.')

    @wraps(fn)
    def getter_fn(*args, **kwargs):
      canonicalized_args = list(args)

      if 'get' in kwargs:
        get_is_not_tuple, get = canonicalize_get(kwargs['get'])
        kwargs['get'] = get
      elif get_index < len(args):
          get_is_not_tuple, get = canonicalize_get(args[get_index])
          canonicalized_args[get_index] = get
      elif defaults is None:
        raise ValueError(
            '`get_namedtuple` function must have a `get` argument provided or'
            'set by default.')
      else:
        get_is_not_tuple, get = canonicalize_get(
            defaults[get_index - len(args)])

      fn_out = fn(*canonicalized_args, **kwargs)

      if get_is_not_tuple:
        if isinstance(fn_out, types.GeneratorType):
          return (output[get[0]] for output in fn_out)
        else:
          return fn_out[get[0]]

      ReturnType = named_tuple_factory(name, get)
      if isinstance(fn_out, types.GeneratorType):
        return (ReturnType(*tuple(output[g] for g in get))
                for output in fn_out)
      else:
        return ReturnType(*tuple(fn_out[g] for g in get))

    return getter_fn
  return getter_decorator
