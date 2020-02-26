from __future__ import print_function

from data_generation_CAE import generate_sample, data_read

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import seaborn as sns

import tensorflow as tf
from tensorflow.contrib import rnn

filepre1 = "pre1.txt"
fileobjpre1 = open(filepre1, mode='w')

filegt = "gt.txt"
fileobjgt = open(filegt, mode='w')

np.random.seed(7)
tf.set_random_seed(7)

# Reading data
Xread, Yread, array_zero = data_read()
# plt.plot(np.transpose(np.array(Yread)))
# plt.show()

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 50
display_step = 10

# Network Parameters
n_input = 16  # input is sin(x), a scalar
n_steps = 10  # timesteps 5
n_hidden = 512   # hidden layer num of features
n_outputs = 302  # output is a series of sin(x+...)
n_layers = 3  # number of stacked LSTM layers 3

loss_array = []
loss_array_test = []

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_outputs]))
}

# Define the LSTM cells
lstm_cells = [rnn.LSTMCell(n_hidden, forget_bias=1.0) for _ in range(n_layers)]
stacked_lstm = rnn.MultiRNNCell(lstm_cells)
    
# lstm_cell = rnn.LSTMCell(n_hidden, forget_bias = 1.0)    
# lstm_dropout = [rnn.LayerNormBasicLSTMCell(n_hidden, forget_bias = 1.0) for _ in range(n_layers)]
# stacked_lstm = rnn.MultiRNNCell(lstm_dropout)

outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=x, dtype=tf.float32, time_major=False)

h = tf.transpose(outputs, [1, 0, 2])
pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    loss_value = None
    target_loss = 100

    # Keep training until we reach max iterations
    while step * batch_size < training_iters:

        _, _, batch_x, batch_y, _, _, _, _, _, _, test_x, test_y= generate_sample(0, X_read = Xread, Y_read = Yread,f=None, t0=None, batch_size=batch_size,
                                                  samples=n_steps, predict=1, ninputs = n_input, noutputs = n_outputs)

        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size, n_outputs))

        test_x = test_x.reshape((batch_size, n_steps, n_input))
        test_y = test_y.reshape((batch_size, n_outputs))

        # Run optimization op (back propagation)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss

            loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})

            loss_value_test = 0
            test_size = 5
            for i in range(test_size):
                _, _, batch_x, batch_y, _, _, _, _, _, _, test_x, test_y= generate_sample(i*15, X_read = Xread, Y_read = Yread,f=None, t0=None, batch_size=batch_size,
                              samples=n_steps, predict=1, ninputs = n_input, noutputs = n_outputs)

                test_x = test_x.reshape((batch_size, n_steps, n_input))
                test_y = test_y.reshape((batch_size, n_outputs))

                loss_value_test += sess.run(loss, feed_dict={x: test_x, y: test_y})
            loss_value_test = loss_value_test / test_size
            loss_array.append(loss_value)
            loss_array_test.append(loss_value_test)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss_value), ", Test Loss= " +"{:.6f}".format(loss_value_test))
        step += 1
    print("Optimization Finished!")

    plt.plot(loss_array)
    plt.plot(loss_array_test)
    plt.ylim(0, 500)
    print(len(loss_array))
    title_array = ['Epochs: ', str(training_iters), ', Layers: ', str(n_layers), ', Hidden layer size: ', str(n_hidden), ', Timesteps: ', str(n_steps)]
    title = ''
    plt.title(title.join(title_array), fontsize = 9)
    # plt.text(len(loss_array), 400, "Number of epochs"+str(training_iters), fontsize=12)
    plt.show()

    gt = []
    pre = []
    # Test the prediction
    for i in range(100-n_steps):
        _, _, _, _, _, _, _, _, _, _, x_pred, y_ground_truth = generate_sample(i, X_read = Xread, Y_read = Yread, f=i, t0=None, samples=n_steps, predict=1, ninputs = n_input, noutputs = n_outputs)

        test_input = x_pred.reshape((1, n_steps, n_input))
        prediction = sess.run(pred, feed_dict={x: test_input})

        print(test_input)

        # remove the batch size dimensions
        prediction = prediction.squeeze()
        y_ground_truth = y_ground_truth.squeeze()

        gt.append(y_ground_truth)
        pre.append(prediction)

    gt = np.array(gt)
    pre = np.array(pre)

    print(gt.shape, pre.shape)



    plt.rcParams["figure.figsize"] = [3.5, 1.5]
    

    plt.subplot(311)
    plt.plot(gt[:,10], color = 'gray')
    plt.plot(pre[:,10], color = '#bf1b2c')
    plt.subplot(312)
    plt.plot(gt[:,45], color = 'gray')
    plt.plot(pre[:,45], color = '#bf1b2c')
    plt.ylabel("Output features")
    plt.subplot(313)
    plt.plot(gt[:,76], color = 'gray')
    plt.plot(pre[:,76], color = '#bf1b2c')
    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.savefig("rnn_cae_result.pdf")
    
    plt.show()

for i in range(pre.shape[0]):
    zzz = 0
    for j in range(324):
        if j in array_zero:
            fileobjpre1.write(str(0.0)+' ')
            zzz += 1
        else:
            fileobjpre1.write(str(pre[i][j-zzz])+' ')
    fileobjpre1.write('\n')

for i in range(pre.shape[0]):
    zzz = 0
    for j in range(324):
        if j in array_zero:
            fileobjgt.write(str(0.0)+' ')
            zzz += 1
        else:
            fileobjgt.write(str(gt[i][j-zzz])+' ')
    fileobjgt.write('\n')