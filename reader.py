# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf


class PTBInput(object):
    """The input data."""
    def __init__(self, config, data, label, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.epoch_size = (len(data) // batch_size)
        self.input_data, self.targets, self.data_init, self.label_init, self.data_ph, self.label_ph, \
            self.data_vb, self.label_vb = ptb_producer(data, label, batch_size, config, name=name)


def ptb_producer(raw_data, raw_label, batch_size, config, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

    Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

    Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, raw_label, batch_size]):
        feature_len = raw_data.shape[1]
        class_len = raw_label.shape[1]
        data_len = raw_data.shape[0]
        epoch_size = data_len // batch_size

        with tf.device(config.use_gpu):
            data_ph = tf.placeholder(dtype=tf.float32, shape=raw_data.shape, name='data_ph')
            label_ph = tf.placeholder(dtype=tf.float32, shape=raw_label.shape, name='label_ph')
            data_vb = tf.Variable(data_ph, trainable=False, collections=[], name='data_vb')
            label_vb = tf.Variable(label_ph, trainable=False, collections=[], name='label_vb')

            #         epoch_size = batch_len
            #         assertion = tf.assert_positive(
            #             epoch_size,
            #             message="epoch_size == 0, decrease batch_size or num_steps")
            # with tf.control_dependencies([assertion]):
            #     epoch_size = tf.identity(epoch_size, name="epoch_size")

            # image, label = tf.train.slice_input_producer([data_vb, label_vb], name='producer')
            # x, y = tf.train.batch([image, label], batch_size=batch_size, capacity=int(batch_size * (0.4 * epoch_size + 3)), name='batch')

            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            x = tf.strided_slice(data_vb, [i * batch_size, 0], [(i + 1) * batch_size, feature_len])
            x.set_shape([batch_size, feature_len])
            y = tf.strided_slice(label_vb, [i * batch_size, 0], [(i + 1) * batch_size, class_len])
            y.set_shape([batch_size, class_len])
    return x, y, data_vb.initializer, label_vb.initializer, data_ph, label_ph, data_vb, label_vb
