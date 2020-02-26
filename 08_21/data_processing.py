import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal
from sklearn import preprocessing

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

print(timeseries_X_scaled[0])


# plt.plot(timeseries_X_scaled[0])
# plt.show()

# Skinflow

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

print(len(content_numerical))

timeseries_X= [[0 for i in range(len(content_numerical))] for j in range(len(content_numerical[0]))]

for i in range(len(timeseries_X)):
	for j in range(len(timeseries_X[0])):
		timeseries_X[i][j] = content_numerical[j][i]

timeseries_X_scaled = []

for items in timeseries_X:
	timeseries_X_scaled.append(preprocessing.scale(items))




