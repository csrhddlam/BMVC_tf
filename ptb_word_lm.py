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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime

import numpy as np
import tensorflow as tf

from reader import PTBInput
from data_loader import get_data
from model import PTBModel
import configs
import operator
from sklearn.metrics import f1_score
import signal


def sigterm_handler(signum, frame):
    print("Received Signal: %s at frame: %s" % (signum, frame))

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "train", "A type of model. Possible options are: train, test.")
flags.DEFINE_string("data_path", '/mnt/4T-HD/why/simple-examples/data', 'Where the training/test data is stored.')
flags.DEFINE_string("save_path", 'save', "Model output directory.")

FLAGS = flags.FLAGS


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""

    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "logits": model.logits,
        "targets": model.targets,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    inputs = []
    outputs = []

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=session, coord=coord)
    # try:
    #     step = 0
    #     while not coord.should_stop():
    for step in range(model.input.epoch_size):
        vals = session.run(fetches)
        cost = vals["cost"]
        state = vals["final_state"]
        targets = vals["targets"]
        logits = vals["logits"]
        for i in range(len(logits)):
            index, value = max(enumerate(targets[i]), key=operator.itemgetter(1))
            inputs.append(index)

        for i in range(len(logits)):
            index, value = max(enumerate(logits[i]), key=operator.itemgetter(1))
            outputs.append(index)
        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f loss: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, costs / iters,
                   iters * model.input.batch_size / (time.time() - start_time)))
        # step += 1
    # except tf.errors.OutOfRangeError:
    #     print('Saving')
    #     saver.save(sess, FLAGS.train_dir, global_step=step)
    #     print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    # finally:
    #     coord.request_stop()
    # coord.join(threads)
    # session.close()

    f1 = f1_score(outputs, inputs, average='macro')
    confusion_matrix = np.zeros((40, 40))
    for i in range(len(inputs)):
        confusion_matrix[inputs[i], outputs[i]] += 1
    confusion_matrix = confusion_matrix[0:39, 0:40]

    temp = np.copy(confusion_matrix)
    for i in range(39):
        temp[i, i] = 0
    # error1 = (np.sum(confusion_matrix) - np.sum(np.diag(confusion_matrix))) / np.sum(confusion_matrix)
    error = np.sum(temp) / np.sum(confusion_matrix)
    return costs / iters, error, f1


def get_config():
    if configs.which_config == 0:
        return configs.Config0()
    elif configs.which_config == 1:
        return configs.Config1()
    elif configs.which_config == 2:
        return configs.Config2()
    else:
        return configs.Default()


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    back_samples = 0
    [for_data, label, index_list, back_data] = get_data(range(39), back_samples, 7, 176, 0.65)
    all_data = np.transpose(np.concatenate((for_data, back_data), axis=1))
    all_gt = np.concatenate((label, 40 * np.ones((1, back_samples), dtype=np.int32)), axis=1)[0] - 1
    # all_data = np.random.randn(7777, 1000)
    # all_gt = np.random.randint(0, 40, 7777)

    all_label = np.zeros((all_gt.shape[0], 40))
    for i in range(all_gt.shape[0]):
        all_label[i, all_gt[i]] = 1

    partition = np.random.randint(0, 5, len(all_label))
    train_data = all_data[partition <= 2]
    train_label = all_label[partition <= 2]
    valid_data = all_data[partition == 3]
    valid_label = all_label[partition == 3]
    test_data = all_data[partition == 4]
    test_label = all_label[partition == 4]

    config = get_config()
    eval_config = get_config()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, label=train_label, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, label=valid_label, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, label=test_label, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
        tf_config.allow_soft_placement = True

        # sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        # with sv.managed_session(config=tf_config) as session:
        with tf.Session(config=tf_config) as session:

            train_writer = tf.summary.FileWriter('summary')
            train_writer.add_graph(tf.get_default_graph())

            session.run(m.init1)
            session.run(m.init2)
            session.run(mvalid.init1)
            session.run(mvalid.init2)
            session.run(mtest.init1)
            session.run(mtest.init2)
            session.run(m.input.data_init, feed_dict={m.input.data_ph: train_data})
            session.run(m.input.label_init, feed_dict={m.input.label_ph: train_label})
            session.run(mvalid.input.data_init, feed_dict={mvalid.input.data_ph: valid_data})
            session.run(mvalid.input.label_init, feed_dict={mvalid.input.label_ph: valid_label})
            session.run(mtest.input.data_init, feed_dict={mtest.input.data_ph: test_data})
            session.run(mtest.input.label_init, feed_dict={mtest.input.label_ph: test_label})

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)

            for i in range(config.max_max_epoch):
                # temp1 = tf.Graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                # temp2 = tf.Graph.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.7f" % (i + 1, session.run(m.lr)))

                train_loss, train_error, train_f1 = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Loss: %.7f Train Error: %.7f Train F1: %.7f" % (i + 1, train_loss, train_error, train_f1))
                valid_loss, valid_error, valid_f1 = run_epoch(session, mvalid)
                print("Epoch: %d Valid Loss: %.7f Valid Error: %.7f Valid F1: %.7f" % (i + 1, valid_loss, valid_error, valid_f1))

            test_loss, test_error, test_f1 = run_epoch(session, mtest)
            print("Test Loss: %.7f Test Error: %.7f Test F1: %.7f" % (test_loss, test_error, test_f1))

            # coord.request_stop()
            # coord.join(threads)

            # if FLAGS.save_path:
            #     print("Saving model to %s." % FLAGS.save_path)
                # sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
    tf.app.run()
