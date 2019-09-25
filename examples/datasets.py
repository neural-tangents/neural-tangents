# Copyright 2019 The Neural Tangents Authors.  All rights reserved.

"""Datasets used in examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax import random
import numpy as np
import tensorflow_datasets as tfds


def _partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def mnist(n_train=None, n_test=None, permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  ds_train, ds_test = tfds.as_numpy(
      tfds.load(
          "mnist",
          split=["train", "test"],
          batch_size=-1,
          as_dataset_kwargs={"shuffle_files": False}))
  train_images, train_labels, test_images, test_labels = (ds_train["image"],
                                                          ds_train["label"],
                                                          ds_test["image"],
                                                          ds_test["label"])

  train_images = _partial_flatten_and_normalize(train_images)
  test_images = _partial_flatten_and_normalize(test_images)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if n_train is not None:
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
  if n_test is not None:
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def minibatch(x_train, y_train, batch_size, train_epochs):
  """Generate minibatches of data for a set number of epochs."""
  epoch = 0
  start = 0
  key = random.PRNGKey(0)

  while epoch < train_epochs:
    end = start + batch_size

    if end > x_train.shape[0]:
      key, split = random.split(key)
      permutation = random.shuffle(split,
                                   np.arange(x_train.shape[0], dtype=np.int64))
      x_train = x_train[permutation]
      y_train = y_train[permutation]
      epoch = epoch + 1
      start = 0
      continue

    yield x_train[start:end], y_train[start:end]
    start = start + batch_size
