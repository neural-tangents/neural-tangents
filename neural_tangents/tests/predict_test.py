# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""Tests for `utils/predict.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import test_util as jtu
from jax.api import grad
from jax.api import jit
from jax.config import config
from jax.experimental import optimizers
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import predict
from neural_tangents import stax
from neural_tangents.utils import empirical


config.parse_flags_with_absl()


MATRIX_SHAPES = [(3, 3), (4, 4)]
OUTPUT_LOGITS = [1, 2, 3]

GETS = ('ntk', 'nngp', ('ntk', 'nngp'))

RTOL = 0.1
ATOL = 0.1

if not config.read('jax_enable_x64'):
  RTOL = 0.2
  ATOL = 0.2


FLAT = 'FLAT'
POOLING = 'POOLING'

TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3)]
NETWORK = [FLAT, FLAT, FLAT, FLAT]
OUTPUT_LOGITS = [1, 2, 3]

CONVOLUTION_CHANNELS = 256


def _build_network(input_shape, network, out_logits):
  if len(input_shape) == 1:
    assert network == 'FLAT'
    return stax.serial(
        stax.Dense(4096, W_std=1.2, b_std=0.05),
        stax.Erf(),
        stax.Dense(out_logits, W_std=1.2, b_std=0.05))
  elif len(input_shape) == 3:
    if network == 'POOLING':
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), stax.Dense(out_logits, W_std=2.0, b_std=0.05))
    elif network == 'FLAT':
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.Flatten(), stax.Dense(out_logits, W_std=2.0, b_std=0.05))
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _empirical_kernel(key, input_shape, network, out_logits):
  init_fn, f, _ = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  _kernel_fn = empirical.empirical_kernel_fn(f)
  kernel_fn = lambda x1, x2, get: _kernel_fn(x1, x2, params, get)
  return params, f, jit(kernel_fn, static_argnums=(2,))


def _theoretical_kernel(key, input_shape, network, out_logits):
  init_fn, f, kernel_fn = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  return params, f, jit(kernel_fn, static_argnums=(2,))


KERNELS = {
    'empirical': _empirical_kernel,
    'theoretical': _theoretical_kernel,
}


@optimizers.optimizer
def momentum(learning_rate, momentum=0.9):
  """A standard momentum optimizer for testing.

  Different from `jax.experimental.optimizers.momentum` (Nesterov).
  """
  learning_rate = optimizers.make_schedule(learning_rate)
  def init_fn(x0):
    v0 = np.zeros_like(x0)
    return x0, v0
  def update_fn(i, g, state):
    x, velocity = state
    velocity = momentum * velocity + g
    x = x - learning_rate(i) * velocity
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  return init_fn, update_fn, get_params


class PredictTest(jtu.JaxTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train={}_test={}_network={}_logits={}_{}'.format(
              train, test, network, out_logits, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'out_logits': out_logits,
          'fn_and_kernel': fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testNTKMSEPrediction(
      self, train_shape, test_shape, network, out_logits, fn_and_kernel):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, train_shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda params, x: \
        0.5 * np.mean((f(params, x) - data_labels) ** 2)
    grad_loss = jit(grad(loss))

    g_dd = ntk(data_train, None, 'ntk')
    g_td = ntk(data_test, data_train, 'ntk')

    predictor = predict.gradient_descent_mse(g_dd, data_labels, g_td)

    atol = ATOL
    rtol = RTOL
    step_size = 0.1

    if len(train_shape) > 2:
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        0.0, fx_initial_train, fx_initial_test)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        train_time, fx_initial_train, fx_initial_test)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), True, rtol, atol)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), True, rtol, atol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train={}_test={}_network={}_logits={}_{}'.format(
              train, test, network, out_logits, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'out_logits': out_logits,
          'fn_and_kernel': fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testNTKGDPrediction(
      self, train_shape, test_shape, network, out_logits, fn_and_kernel):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, train_shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat) ** 2)
    grad_loss = jit(grad(lambda params, x: loss(f(params, x), data_labels)))

    g_dd = ntk(data_train, None, 'ntk')
    g_td = ntk(data_test, data_train, 'ntk')

    predictor = predict.gradient_descent(g_dd, data_labels, loss, g_td)

    atol = ATOL
    rtol = RTOL
    step_size = 0.5

    if len(train_shape) > 2:
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        0.0, fx_initial_train, fx_initial_test)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        train_time, fx_initial_train, fx_initial_test)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), True, rtol, atol)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), True, rtol, atol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train={}_test={}_network={}_logits={}_{}'.format(
              train, test, network, out_logits, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'out_logits': out_logits,
          'fn_and_kernel': fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()
                          if len(train) == 2))
  def testNTKMomentumPrediction(
      self, train_shape, test_shape, network, out_logits, fn_and_kernel):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, train_shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat) ** 2)
    grad_loss = jit(grad(lambda params, x: loss(f(params, x), data_labels)))

    g_dd = ntk(data_train, None, 'ntk')
    g_td = ntk(data_test, data_train, 'ntk')

    atol = ATOL
    rtol = RTOL
    step_size = 0.5

    if len(train_shape) > 2:
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / np.sqrt(step_size))

    init, predictor, get = predict.momentum(
        g_dd, data_labels, loss, step_size, g_td)

    opt_init, opt_update, get_params = momentum(step_size, 0.9)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    lin_state = init(fx_initial_train, fx_initial_test)
    fx_pred_train, fx_pred_test = get(lin_state)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    lin_state = predictor(lin_state, train_time)
    fx_pred_train, fx_pred_test = get(lin_state)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), True, rtol, atol)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), True, rtol, atol)


  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}'.format(
                  train, test, network, out_logits),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
      }
                          for train, test, network in zip(
                              TRAIN_SHAPES[:-1], TEST_SHAPES[:-1], NETWORK[:-1])
                          for out_logits in OUTPUT_LOGITS))

  def testNTKMeanPrediction(
      self, train_shape, test_shape, network, out_logits):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = np.cos(random.normal(split, test_shape))
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)
    mean_pred, var = predict.gp_inference(kernel_fn, data_train, data_labels,
                                          data_test, 'ntk', diag_reg=0.,
                                          compute_var=True)

    if xla_bridge.get_backend().platform == 'tpu':
      eigh = np.onp.linalg.eigh
    else:
      eigh = np.linalg.eigh

    self.assertEqual(var.shape[0], data_test.shape[0])
    min_eigh = np.min(eigh(var)[0])
    self.assertGreater(min_eigh + 1e-10, 0.)
    def mc_sampling(count=10):
      empirical_mean = 0.
      key = random.PRNGKey(100)
      init_fn, f, _ = _build_network(train_shape[1:], network, out_logits)
      _kernel_fn = empirical.empirical_kernel_fn(f)
      kernel_fn = jit(lambda x1, x2, params: _kernel_fn(x1, x2, params, 'ntk'))

      for _ in range(count):
        key, split = random.split(key)
        _, params = init_fn(split, train_shape)

        g_dd = kernel_fn(data_train, None, params)
        g_td = kernel_fn(data_test, data_train, params)
        predictor = predict.gradient_descent_mse(g_dd, data_labels, g_td)

        fx_initial_train = f(params, data_train)
        fx_initial_test = f(params, data_test)

        _, fx_pred_test = predictor(1.0e8, fx_initial_train, fx_initial_test)
        empirical_mean += fx_pred_test
      return empirical_mean / count
    atol = ATOL
    rtol = RTOL
    mean_emp = mc_sampling(100)

    self.assertAllClose(mean_pred, mean_emp, True, rtol, atol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}'.format(
                  train, test, network, out_logits),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
      }
                          for train, test, network in zip(
                              TRAIN_SHAPES[:-1], TEST_SHAPES[:-1], NETWORK[:-1])
                          for out_logits in OUTPUT_LOGITS))

  def testGPInferenceGet(
      self, train_shape, test_shape, network, out_logits):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = np.cos(random.normal(split, test_shape))
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

    out = predict.gp_inference(kernel_fn, data_train, data_labels,
                               data_test, 'ntk', diag_reg=0.,
                               compute_var=True)
    assert isinstance(out, predict.Gaussian)

    out = predict.gp_inference(kernel_fn, data_train, data_labels,
                               data_test, 'nngp', diag_reg=0.,
                               compute_var=True)
    assert isinstance(out, predict.Gaussian)

    out = predict.gp_inference(kernel_fn, data_train, data_labels,
                               data_test, ('ntk',), diag_reg=0.,
                               compute_var=True)
    assert len(out) == 1 and isinstance(out[0], predict.Gaussian)

    out = predict.gp_inference(kernel_fn, data_train, data_labels,
                               data_test, ('ntk', 'nngp'), diag_reg=0.,
                               compute_var=True)
    assert (len(out) == 2 and
            isinstance(out[0], predict.Gaussian) and
            isinstance(out[1], predict.Gaussian))

    out2 = predict.gp_inference(kernel_fn, data_train, data_labels,
                               data_test, ('nngp', 'ntk'), diag_reg=0.,
                               compute_var=True)
    self.assertAllClose(out[0], out2[1], True)
    self.assertAllClose(out[1], out2[0], True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}_get={}'.format(
                  train, test, network, out_logits, get),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
          'get':
              get,
      }
                          for train, test, network in zip(
                              TRAIN_SHAPES[:-1], TEST_SHAPES[:-1], NETWORK[:-1])
                          for out_logits in OUTPUT_LOGITS
                          for get in GETS))
  def testInfiniteTimeAgreement(
      self, train_shape, test_shape, network, out_logits, get):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = np.cos(random.normal(split, test_shape))
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    inf_prediction = predict.gp_inference(
        kernel_fn, data_train, data_labels, data_test, get,
        diag_reg=reg, compute_var=True)
    prediction = predict.gradient_descent_mse_gp(
        kernel_fn, data_train, data_labels, data_test, get,
        diag_reg=reg, compute_var=True)

    finite_prediction = prediction(np.inf)

    self.assertAllClose(inf_prediction, finite_prediction, True)


if __name__ == '__main__':
  jtu.absltest.main()
