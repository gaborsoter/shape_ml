import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal

plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 7
fig = plt.figure(figsize=(4, 6)) 
ax = fig.add_subplot(111)

fs = 20
fc = 2  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
# reading data starts -----------------------

content = open("data/08_14/lines/image_processing_skinflow_lines.txt", "r").read().splitlines()

print('content_length:', len(content))

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


content_numerical_time = [[0 for i in range(len(content_numerical))] for j in range(len(content_numerical[0]))]

for i in range(len(content_numerical_time)):
	for j in range(len(content_numerical_time[0])):
		content_numerical_time[i][j] = content_numerical[j][i]

# # reading data ends -------------------------

# for i in range(100):
# 	for time in range(20):
# 		baselines_x[i] += content_numerical[time][i][0]
# 		baselines_y[i] += content_numerical[time][i][1]
# 	baselines_x[i] = baselines_x[i] / 20
# 	baselines_y[i] = baselines_y[i] / 20

# for i in range(100):
# 	for time in range(len(content_numerical)):
# 		content_numerical[time][i][0] = content_numerical[time][i][0]-baselines_x[i]
# 		content_numerical[time][i][1] = content_numerical[time][i][1]-baselines_y[i]

# # baseline calculation ends -----------------

# # visualisation starts ----------------------

# x_plot_3 = []
# y_plot_3 = []
# x_plot_50 = []
# y_plot_50 = []
# x_plot_76 = []
# y_plot_76 = []



# for time in range(len(content_numerical)):
	# x_plot_3.append(content_numerical[time][4][0])
	# y_plot_3.append(content_numerical[time][4][1])
	# x_plot_50.append(content_numerical[time][50][0])
	# y_plot_50.append(content_numerical[time][50][1])
	# x_plot_76.append(content_numerical[time][76][0])
	# y_plot_76.append(content_numerical[time][76][1])

# # filtering starts --------------------------
# b, a = signal.butter(5, w, 'low')
# x_plot_3_f = signal.filtfilt(b, a, x_plot_3)
# y_plot_3_f = signal.filtfilt(b, a, y_plot_3)
# x_plot_50_f = signal.filtfilt(b, a, x_plot_50)
# y_plot_50_f = signal.filtfilt(b, a, y_plot_50)
# x_plot_76_f = signal.filtfilt(b, a, x_plot_76)
# y_plot_76_f = signal.filtfilt(b, a, y_plot_76)
# # filtering ends   --------------------------

ax.set_xlim([0, 500])
# ax.set_ylim([78, 80])
ax.set_xlabel('Frame number', fontsize=7)
ax.set_ylabel('Horizontal marker position', fontsize=7)
# ax.set_aspect(aspect=20)
plt.grid(False)
ax.tick_params(direction='out', length=2, width=1, which='both', axis='both')

# ax2 = fig.add_subplot(212)

# ax2.set_xlim([0, 500])
# ax2.set_ylim([-10, 10])
# ax2.set_xlabel('Frame number', fontsize=7)
# ax2.set_ylabel('Vertical marker position', fontsize=7)
# ax2.set_aspect(aspect=20)
# plt.grid(False)
# ax2.tick_params(direction='out', length=2, width=1, which='both', axis='both')

b, a = signal.butter(1, w, 'low')
ax.plot(content_numerical_time[7], color = '#e00045')
x_plot_3_f = signal.filtfilt(b, a, content_numerical_time[7])
ax.plot(x_plot_3_f, color = '#000000')
# ax.plot(content_numerical_time[14], color = '#00288e')

# ax2.plot(x_plot_3_f, color = '#00288e')
# ax2.plot(x_plot_76_f, color = '#e08e00')
# ax2.plot(x_plot_50_f, color = '#e00045')

fig.savefig('skinflow_filtered.png')
plt.show()
# visualisation ends -------------------------