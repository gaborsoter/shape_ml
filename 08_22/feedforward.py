import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
from scipy import signal
from sklearn import preprocessing
import sys

np.set_printoptions(threshold=sys.maxsize)

# Pointcloud

content = open("../data/image_processing_dots_lines.txt", "r").read().splitlines()

content_numerical = []
for t in range(len(content)):
	timestep = []
	for i in range(len(content[t].split('],'))):
		try:
			line = [int(content[t].split('],')[i].split(',')[0].split('[[')[1]), int(content[t].split('],')[i].split(',')[1])]
		except:
			try:
				line = [int(content[t].split('],')[i].split(',')[0].split('[')[1]), int(content[t].split('],')[i].split(',')[1])]
			except:
				line = [int(content[t].split('],')[i].split(',')[0].split('[')[1]), int(content[t].split('],')[i].split(',')[1].split(']]')[0])]
		timestep.append(line)
	content_numerical.append(timestep)

timeseries_Y = [[0 for t in range(len(content_numerical))] for i in range(100)]

for time in range(len(content_numerical)):
	for i in range(100):
		timeseries_Y[i][time] = content_numerical[time][i]

timeseries_Y_scaled = []

for items in timeseries_Y:
	timeseries_Y_scaled.append(preprocessing.scale(items))

timeseries_Y_scaled = np.array(timeseries_Y_scaled)
# Skinflow scaled

content = open("../data/image_processing_skinflow_lines.txt", "r").read().splitlines()
content_numerical = []

for time in range(len(content)):
	line = []
	for i in range(16):
		if i == 0:
			line.append(int(content[time].split(',')[0].split('[')[1]))
		elif i == 15 :
			line.append(int(content[time].split(',')[15].split(']')[0]))
		else:
			line.append(int(content[time].split(',')[i]))
	content_numerical.append(line)

timeseries_X= [[0 for i in range(len(content_numerical))] for j in range(len(content_numerical[0]))]

for i in range(len(timeseries_X)):
	for j in range(len(timeseries_X[0])):
		timeseries_X[i][j] = content_numerical[j][i]

timeseries_X_scaled = []

for items in timeseries_X:
	timeseries_X_scaled.append(preprocessing.scale(items))


with tf.name_scope('placeholders'):
    x = tf.placeholder('float', [None, 200])
    y = tf.placeholder('float', [None, 16])

with tf.name_scope('neural_network'):
    x1 = tf.contrib.layers.fully_connected(x, 1028, activation_fn=tf.nn.relu)
    x2 = tf.contrib.layers.fully_connected(x1, 512, activation_fn=tf.nn.relu)
    x3 = tf.contrib.layers.fully_connected(x2, 256, activation_fn=tf.nn.relu)
    result = tf.contrib.layers.fully_connected(x3, 16, activation_fn=None)

    loss = tf.nn.l2_loss(result - y)

with tf.name_scope('optimizer'):
    train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")

sess.run(tf.global_variables_initializer())

batch_size = 100
difficultshit_flattened = np.array([[0 for j in range(200)] for i in range(100)])

loss_array = []
loss_test_array = []

# Train the network
for i_epoch in range(1000): #100
	print(i_epoch, " / 200")
	for i_batch in range(41): # 42 fix
		realshit = np.transpose(timeseries_X_scaled)[i_batch*batch_size:i_batch*batch_size+batch_size]
		difficultshit = np.swapaxes(timeseries_Y_scaled, 0, 1)[i_batch*batch_size:i_batch*batch_size+batch_size]
		for time in range(difficultshit.shape[0]):
			difficultshit_flattened[time] = difficultshit[time].flatten()
		_, loss_result = sess.run([train_op, loss],
	                              feed_dict={y: realshit[:, :],
	                                         x: difficultshit_flattened[:, :]})

		realshit = np.transpose(timeseries_X_scaled)[41*100:41*100+100]
		difficultshit = np.swapaxes(timeseries_Y_scaled, 0, 1)[41*100:41*100+100]
		_, loss_test_result = sess.run([train_op, loss],
	                              feed_dict={y: realshit[:, :],
	                                         x: difficultshit_flattened[:, :]})

	print("Training loss: ", loss_result, "Test loss:", loss_test_result)
	loss_array.append(loss_result)
	loss_test_array.append(loss_test_result)

# Evaluation

realshit = np.transpose(timeseries_X_scaled)[41*100:41*100+100]
difficultshit = np.swapaxes(timeseries_Y_scaled, 0, 1)[41*100:41*100+100]
for time in range(difficultshit.shape[0]):
	difficultshit_flattened[time] = difficultshit[time].flatten()

result = sess.run(result, feed_dict = {y: realshit[:, :], x: difficultshit_flattened[:, :]})
plt.plot(difficultshit_flattened[:, 10:13],)
plt.plot(result[:, 10:13], markersize=3)

plt.show()

plt.plot(loss_array)
plt.plot(loss_test_array)
plt.show()
