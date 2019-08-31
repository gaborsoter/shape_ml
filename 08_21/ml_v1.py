import numpy as np
import tensorflow as tf 
from tensorflow.python import debug as tf_debug

k_true = [[1, -1], [3, -3], [2, -2]]
b_true = [-5, 5]
num_examples = 120

with tf.Session() as sess:
	x = tf.placeholder(tf.float32, shape = [None, 3], name = "x")
	y = tf.placeholder(tf.float32, shape = [None, 2], name = "y")

	dense_layer = tf.keras.layers.Dense(2, use_bias = True)
	y_hat = dense_layer(x)
	loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_hat), name = "loss")
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	sess.run(tf.global_variables_initializer())
	sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")

	for i in range(50):
		xs = np.random.randn(num_examples, 3)
		ys = np.matmul(xs, k_true) + b_true

		loss_val, _ = sess.run([loss, optimizer], feed_dict = {x: xs, y: ys})
		print("Iteration: %d, loss: %g" % (i, loss_val))