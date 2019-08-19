import numpy as np
import cv2
import time

cap = cv2.VideoCapture('data/out2_2019_09_16_36.mp4')

while(True):
    ret, frame = cap.read()

    frame = frame[30:200, 75:245]
    blur = cv2.GaussianBlur(frame, (int(5), int(5)), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) # grayscale
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,1001,0)

    # thresh = 255 - thresh

    kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(thresh,kernel,iterations = 1)
    erosion = cv2.erode(thresh,kernel,iterations = 1)

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
            cv2.circle(frame, centres[-1], 2, (0, 255, 0), -1)


    cv2.imshow('frame', frame)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()