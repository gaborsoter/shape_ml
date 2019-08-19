import numpy as np
import cv2
import time

cap = cv2.VideoCapture('data/08_14/random/batch_1000_1.avi')

cv2.namedWindow('image')

def nothing(x):
    pass

cv2.createTrackbar('thresh1','image',3,1001,nothing)
cv2.createTrackbar('erosion','image',0,101,nothing)


while(True):
    thresh1  = cv2.getTrackbarPos('thresh1','image')
    thresh2  = cv2.getTrackbarPos('erosion','image')

    if thresh1 % 2 == 0:
        thresh1 += 1

    ret, frame = cap.read()

    rows, cols, ch = frame.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),2,1)
    frame = cv2.warpAffine(frame,M,(cols,rows))

    frame = frame[60:400, 148:474]
    blur = cv2.GaussianBlur(frame, (int(1), int(1)), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)     # grayscale
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,thresh1,thresh2)

    ret,thresh = cv2.threshold(gray,121,255,cv2.THRESH_BINARY)

    

    kernel = np.ones((3,3),np.uint8)
    
    # #dilation = cv2.dilate(thresh,kernel,iterations = 1)
    erosion = cv2.erode(thresh,kernel,iterations = 3)
    # opening = 255 - opening

    im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[6]
    # cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    centres = []
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if (moments['m00'] == 0 or moments['m00'] == 0):
            mx = 0
        else:
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
            cv2.circle(frame, centres[-1], 6, (0, 255, 0), -1)


    cv2.imshow('image', frame)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()