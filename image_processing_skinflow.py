import numpy as np
import cv2
import time

# video capture
cap = cv2.VideoCapture('data/08_14/lines/batch_1000_2.avi')

cv2.namedWindow('image')

def nothing(x):
    pass

cv2.createTrackbar('R_low','image',0,255,nothing)
cv2.createTrackbar('G_low','image',0,255,nothing)
cv2.createTrackbar('B_low','image',0,255,nothing)
cv2.createTrackbar('R_high','image',0,255,nothing)
cv2.createTrackbar('G_high','image',0,255,nothing)
cv2.createTrackbar('B_high','image',0,255,nothing)

# colour ranges
frame_counter = 0

while(True):

    r_low  = cv2.getTrackbarPos('R_low','image')
    g_low  = cv2.getTrackbarPos('G_low','image')
    b_low  = cv2.getTrackbarPos('B_low','image')
    r_high = cv2.getTrackbarPos('R_high','image')
    g_high = cv2.getTrackbarPos('G_high','image')
    b_high = cv2.getTrackbarPos('B_high','image')
    lower  = np.array([b_low, g_low, r_low], dtype = "uint8")
    upper  = np.array([b_high, g_high, r_high], dtype = "uint8")
    ret, frame = cap.read()

    frame_counter += 1
    #If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 #Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame = frame[0:400, 0:600]


    mask = cv2.inRange(frame, lower, upper) 
    output = cv2.bitwise_and(frame, frame, mask = mask) # colour filtering
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscaling

    thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,41,10)

    rows, cols = gray.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-1.5,1)
    rotated = cv2.warpAffine(gray,M,(cols,rows))
# 154 113 97, RGB
    cv2.imshow('image',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()