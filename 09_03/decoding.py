import numpy as np
np.random.seed(1337)

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

from numpy import *
import cv2

filename1 = "pre1.txt"
datainputback = loadtxt(filename1).tolist()

filename2 = "gt.txt"
datagt = loadtxt(filename2).tolist()

datainput5_in = []
datainput5_gt = []

for item in datainputback:
    line_bejon = np.array(item).reshape(9, 9, 4)
    datainput5_in.append(line_bejon.tolist())


for item in datagt:
    line_bejon = np.array(item).reshape(9, 9, 4)
    datainput5_gt.append(line_bejon.tolist())

fourcc = cv2.VideoWriter_fourcc('X','V', 'I', 'D')
outvideo = cv2.VideoWriter('soft_decoded.avi', fourcc, 20.0, (136,68))

model = load_model('CAE_soft_octopus_backup.h5')

# encode and decode some digits
# note that we take them from the *test* set
 
decoded_imgs = model.predict(np.array(datainput5_in))
decoded_imgs_gt = model.predict(np.array(datainput5_gt))

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 1  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
	# display original
	ax = plt.subplot(2, n, i + 1)
	#plt.imshow(x_test[i].reshape(45, 55))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	#plt.imshow(decoded_imgs[i*15].reshape(45, 50))
	plt.imshow(decoded_imgs[i*8].reshape(68,68))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)



for k in range(1):
	vector = []
	frame = cv2.imread('res_1_4degrees.png')
	roi = frame[100:168,100:236]
	writeout = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # grayscale

cap = cv2.VideoCapture('../../Google\ Drive/Projects/PhD/Pubs_n_Confs/Shape_reconstruction/data/08_14/lines/batch_1000_1.avi')

kkkkkk = 1
imagedata_all  = []
num = 0
while(kkkkkk!=0):
	print('see')
	ret,frame = cap.read() # camera readout
	if (ret == True):

		vector = []
        # rotation
		rows, cols, ch = frame.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),2,1)
		frame = cv2.warpAffine(frame,M,(cols,rows))

        # roi
		frame = frame[60:400, 148:488]
        # removing noise
		blur = cv2.GaussianBlur(frame, (int(3), int(3)), 0)
        # grayscaling
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # thresholding
		ret,thresh = cv2.threshold(gray,121,255,cv2.THRESH_BINARY)
        
        # cleaning with erosion
		kernel = np.ones((3,3),np.uint8)
		erosion = cv2.erode(thresh,kernel,iterations = 2)
		erosion = cv2.resize(gray,None,fx=0.2,fy=0.2)

		if (num >= 4000 and num < 4090):
			imagedata_all.append(vector)
	else:
		kkkkkk = 0
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
print(len(imagedata_all))
for t in range(len(decoded_imgs)):
	im = 255*decoded_imgs[t].reshape(68,68)
	for i in range(68):
		for j in range(68):
			writeout[i,j]=im[i,j]
			# writeout[i,j+68]=imagedata_all[t][i,j]
			a = imagedata_all[t]
	backtorgb = cv2.cvtColor(writeout,cv2.COLOR_GRAY2RGB)
	outvideo.write(backtorgb)
