import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal

plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.size"] = 7
fig = plt.figure(figsize=(4, 6)) 
ax = fig.add_subplot(211)

fs = 20
fc = 2  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
# reading data starts -----------------------

content = open("data/08_14/lines/image_processing_dots_lines.txt", "r").read().splitlines()

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

# reading data ends -------------------------




# baseline calculation starts ---------------
baselines_x = [0 for i in range(100)]
baselines_y = [0 for i in range(100)]


for i in range(100):
	for time in range(20):
		baselines_x[i] += content_numerical[time][i][0]
		baselines_y[i] += content_numerical[time][i][1]
	baselines_x[i] = baselines_x[i] / 20
	baselines_y[i] = baselines_y[i] / 20

for i in range(100):
	for time in range(len(content_numerical)):
		content_numerical[time][i][0] = content_numerical[time][i][0]-baselines_x[i]
		content_numerical[time][i][1] = content_numerical[time][i][1]-baselines_y[i]

# baseline calculation ends -----------------

# visualisation starts ----------------------

x_plot_3 = []
y_plot_3 = []
x_plot_50 = []
y_plot_50 = []
x_plot_76 = []
y_plot_76 = []



for time in range(len(content_numerical)):
	x_plot_3.append(content_numerical[time][4][0])
	y_plot_3.append(content_numerical[time][4][1])
	x_plot_50.append(content_numerical[time][50][0])
	y_plot_50.append(content_numerical[time][50][1])
	x_plot_76.append(content_numerical[time][76][0])
	y_plot_76.append(content_numerical[time][76][1])

# filtering starts --------------------------
b, a = signal.butter(5, w, 'low')
x_plot_3_f = signal.filtfilt(b, a, x_plot_3)
y_plot_3_f = signal.filtfilt(b, a, y_plot_3)
x_plot_50_f = signal.filtfilt(b, a, x_plot_50)
y_plot_50_f = signal.filtfilt(b, a, y_plot_50)
x_plot_76_f = signal.filtfilt(b, a, x_plot_76)
y_plot_76_f = signal.filtfilt(b, a, y_plot_76)
# filtering ends   --------------------------

ax.set_xlim([0, 500])
ax.set_ylim([-10, 10])
ax.set_xlabel('Frame number', fontsize=7)
ax.set_ylabel('Horizontal marker position', fontsize=7)
ax.set_aspect(aspect=20)
plt.grid(False)
ax.tick_params(direction='out', length=2, width=1, which='both', axis='both')

ax2 = fig.add_subplot(212)

ax2.set_xlim([0, 500])
ax2.set_ylim([-10, 10])
ax2.set_xlabel('Frame number', fontsize=7)
ax2.set_ylabel('Vertical marker position', fontsize=7)
ax2.set_aspect(aspect=20)
plt.grid(False)
ax2.tick_params(direction='out', length=2, width=1, which='both', axis='both')

ax.plot(y_plot_50_f, color = '#e00045')
ax.plot(y_plot_76_f, color = '#e08e00')
ax.plot(y_plot_3_f, color = '#00288e')

ax2.plot(x_plot_3_f, color = '#00288e')
ax2.plot(x_plot_76_f, color = '#e08e00')
ax2.plot(x_plot_50_f, color = '#e00045')

# fig.savefig('filtered.png')
plt.show()
# visualisation ends -------------------------