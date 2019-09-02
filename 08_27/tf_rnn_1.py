import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
from scipy import signal
import cv2
import time

# ****************** Data preprocessing *********************

#
#
#
# reading and formatting pointcloud data
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

for t in range(len(content_numerical)):
    for i in range(100):
        timeseries_Y[i][t] = content_numerical[t][i]

timeseries_Y_scaled = []

for items in timeseries_Y:
    # timeseries_Y_scaled.append(preprocessing.scale(items))
    timeseries_Y_scaled.append(items)

timeseries_Y_scaled = np.array(timeseries_Y_scaled)

#
#
#
# reading and formatting Skinflow data
content = open("../data/image_processing_skinflow_lines.txt", "r").read().splitlines()
content_numerical = []

for t in range(len(content)):
    line = []
    for i in range(16):
        if i == 0:
            line.append(int(content[t].split(',')[0].split('[')[1]))
        elif i == 15 :
            line.append(int(content[t].split(',')[15].split(']')[0]))
        else:
            line.append(int(content[t].split(',')[i]))
    content_numerical.append(line)

timeseries_X= [[0 for i in range(len(content_numerical))] for j in range(len(content_numerical[0]))]

for i in range(len(timeseries_X)):
    for j in range(len(timeseries_X[0])):
        timeseries_X[i][j] = content_numerical[j][i]

timeseries_X_scaled = []

for items in timeseries_X:
    timeseries_X_scaled.append(preprocessing.scale(items))

# filtering

# fs = 20
# fc = .5  # Cut-off frequency of the filter
# w = fc / (fs / 2) # Normalize the frequency

# for points in range(len(timeseries_Y_scaled)):# finish this (filtering...)
#     normal = np.copy(np.swapaxes(timeseries_Y_scaled,0, 1)[:,2,0])
#     b, a = signal.butter(1, w, 'low')
#     filtered = signal.filtfilt(b, a, normal)

# N = 5
# filtered = np.convolve(normal, np.ones((N,))/N, mode = "valid")

# plt.plot(normal)
# plt.plot(filtered)
# plt.show()

# ******************* Writing video ********************

# cap = cv2.VideoCapture('../../../Google Drive/Projects/PhD/Pubs_n_Confs/Shape_reconstruction/data/08_14/lines/batch_1000_1.avi')

# i_time = 0
# while(True):
#     ret, frame = cap.read()
#     frame = frame[60:400, 148:485]

#     rows, cols, ch = frame.shape
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),2,1)
#     frame = cv2.warpAffine(frame,M,(cols,rows))

#     for i in range(len(timeseries_Y)):
#         cv2.circle(frame, (timeseries_Y[i][i_time][0], timeseries_Y[i][i_time][1]), 6, (0, 255, 0), -1)
#         # cv2.putText(frame,str(i), (sort_all[i][0], sort_all[i][1] + 2), font, fontScale, fontColor, lineType)

#     i_time += 1

#     time.sleep(0.05)

#     cv2.imshow('Image', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




