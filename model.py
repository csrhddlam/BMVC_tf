from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def data_type():
    return tf.float32


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_
        self.init1 = tf.global_variables_initializer()
        self.init2 = tf.local_variables_initializer()
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        output_size = config.output_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)

        def rnn_cell():
            return tf.contrib.rnn.BasicRNNCell(size, activation=tf.sigmoid)
        attn_cell = rnn_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        # with tf.device("/cpu:0"):
        #     embedding = tf.get_variable(
        #         "embedding", [vocab_size, size], dtype=data_type())
        #     inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        inputs = input_.input_data

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.nn.rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs, state)
                outputs.append(cell_output)

        softmax_w = tf.get_variable(
            "softmax_w", [size, output_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [output_size], dtype=data_type())
        logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
        # logits = outputs[-1]
        # targets = np.zeros((batch_size, output_size))
        # targets[range(batch_size), input_.targets] = 1

        # embedding = tf.get_variable("embedding", [output_size, 1], dtype=data_type())
        # targets = tf.nn.embedding_lookup(embedding, input_.targets)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_.targets, logits=logits)

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self.logits = logits
        self.targets = input_.targets

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op