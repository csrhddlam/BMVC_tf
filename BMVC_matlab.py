import numpy as np
import tensorflow as tf
from data_loader import get_data
import datetime
from tensorflow.python.client import timeline
import scipy.io as sio

visible = 8624
hidden = 1
const_samples = 410
const_iteration = 10
learning_rate = 0.1
one_tenth_step = 0.9
momentum = 0.9
patch_size = 7
total_vc = 176

# [data, label, index_list] = get_data([0])
back_samples = 10000
[data, label, index_list, back_data] = get_data(range(39), back_samples, patch_size, total_vc, 0.65)

sio.savemat('data.mat', {'data': np.concatenate((data, back_data), axis=1),
                         'label': np.concatenate((label, 40 * np.ones((1, back_samples))), axis=1)})

vn_data = data[:, index_list[0]]

with tf.device('/gpu:1'):
    w_vv = tf.random_normal([visible, visible], mean=0, stddev=0.01)
    w_vh = tf.random_normal([visible, hidden], mean=0, stddev=0.01)
    w_hh = tf.random_normal([hidden, hidden], mean=0, stddev=0.01)
    b_v = tf.random_normal([visible, 1], mean=0, stddev=0.01)
    b_h = tf.random_normal([hidden, 1], mean=0, stddev=0.01)

    vn = tf.Variable(vn_data, name='vn')
    W_vv = tf.Variable(w_vv, name='w_vv')
    W_vh = tf.Variable(w_vh, name='w_vh')
    W_hh = tf.Variable(w_hh, name='w_hh')
    B_v = tf.Variable(b_v, name='b_v')
    B_h = tf.Variable(b_h, name='b_h')

    output = []
    # mvn = tf.Variable(b_h, name='b_h')
    mvn = vn
    # output.append(y)
    for i in range(5):
        mvn = tf.add(momentum * mvn, (1 - momentum) * tf.nn.sigmoid(tf.matmul(W_vv, mvn, name='CD1_matmul_' + str(i)) + B_v), name='CD1_mvn_' + str(i))
        # output.append(y)
    # dw_vv = (tf.matmul(vn, vn, transpose_b=True, name='vn_matmul') - tf.matmul(mvn, mvn, transpose_b=True, name='mvn_matmul')) / const_samples

    dw_vv = tf.truediv(tf.subtract(tf.matmul(vn, tf.transpose(vn), name='vn_matmul'),
                                   tf.matmul(mvn, tf.transpose(mvn), name='mvn_matmul'), name='vn2-mvn2'),
                       float(const_samples), name='dw_vv')
    # dw_vv = tf.subtract(tf.matmul(vn, tf.transpose(vn), name='vn_matmul'), tf.matmul(mvn, tf.transpose(mvn), name='mvn_matmul'), name='vn2-mvn2') / const_samples
    db_v = tf.reduce_mean(tf.subtract(vn, mvn, name='vn-mvn'), 1, True, name='db_v')

    output.append(W_vv)
    output.append(B_v)
    output.append(vn)
    output.append(mvn)
    output.append(dw_vv)
    output.append(db_v)

    obs1 = tf.reduce_sum(tf.abs(dw_vv))
    obs2 = tf.reduce_sum(tf.abs(db_v))
    obs3 = tf.reduce_sum(tf.abs(W_vv))
    obs4 = tf.reduce_sum(tf.abs(B_v))

    output.append(obs1)
    output.append(obs2)
    output.append(obs3)
    output.append(obs4)

    # tf.summary.scalar('dw_vv', obs1)
    # tf.summary.scalar('db_v', obs2)

    # sess = tf.InteractiveSession()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True

    sess = tf.InteractiveSession(config=config)
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('summary')

    results = sess.run(output, options=run_options, run_metadata=run_metadata)
    # train_writer.add_graph(tf.get_default_graph())
    # train_writer.add_summary(summary, 0)
    print(results[-4], results[-3], results[-2], results[-1])

    for iteration in range(const_iteration):
        # temp_lr = 0.1 ** (iteration // (one_tenth_step * const_iteration)) * learning_rate
        temp_lr = learning_rate
        op1 = tf.assign_add(W_vv, temp_lr * dw_vv, name='update_w_vv')
        op2 = tf.assign_add(B_v, temp_lr * db_v, name='update_b_v')
        # print(datetime.datetime.now())
        # new_weights = sess.run([op1, op2])
        sess.run([op1, op2])
        print(datetime.datetime.now())
        # results = sess.run(output, options=run_options, run_metadata=run_metadata)
        sess.run(output, options=run_options, run_metadata=run_metadata)

        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

        print(datetime.datetime.now())
        # print(results[-4], results[-3], results[-2], results[-1])

train_writer.add_graph(tf.get_default_graph())

trace = timeline.Timeline(step_stats=run_metadata.step_stats)
trace_file = open('timeline.ctf.json', 'w')
trace_file.write(trace.generate_chrome_trace_format())

