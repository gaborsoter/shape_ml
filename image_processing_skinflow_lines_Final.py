import numpy as np
import cv2
import time

f= open("image_processing_skinflow_lines.txt","w+")
# video capture
cap = cv2.VideoCapture('data/08_14/lines/batch_1000_2.avi')

cv2.namedWindow('image')

def nothing(x):
    pass

def find_first_black(image):
    number = 0
    col, row = image.shape
    for i in range(row):
        if image[0,i] == 0:
            number = i
            break
    return number

cv2.createTrackbar('TP1','image',3,255,nothing)
cv2.createTrackbar('TP2','image',1,260,nothing)



# colour ranges
frame_counter = 0

array_out = []

while(True):
    
    thresh_param_1  = cv2.getTrackbarPos('TP1','image')
    thresh_param_2  = cv2.getTrackbarPos('TP2','image')
    if thresh_param_1 % 2 == 0:
        thresh_param_1 += 1
    if thresh_param_1 == 1:
        thresh_param_1 = 3
    ret, frame = cap.read()

    frame_counter += 1
    #If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0 #Or whatever as long as it is the same as next line
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame = frame[80:330, 140:300]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscaling

    thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,27,12)

    line_p = thresh1[thresh_param_2:thresh_param_2+1, 0:260]
    line1 = thresh1[11:12, 0:260]
    line2 = thresh1[23:24, 0:260]
    line3 = thresh1[39:40, 0:260]
    line4 = thresh1[55:56, 0:260]
    line5 = thresh1[68:69, 0:260]
    line6 = thresh1[83:84, 0:260]
    line7 = thresh1[100:101, 0:260]
    line8 = thresh1[114:115, 0:260]
    line9 = thresh1[129:130, 0:260]
    line10 = thresh1[145:146, 0:260]
    line11 = thresh1[161:162, 0:260]
    line12 = thresh1[175:176, 0:260]
    line13 = thresh1[192:193, 0:260]
    line14 = thresh1[207:208, 0:260]
    line15 = thresh1[221:222, 0:260]
    line16 = thresh1[236:237, 0:260]

    line1_x = find_first_black(line1)
    line2_x = find_first_black(line2)
    line3_x = find_first_black(line3)
    line4_x = find_first_black(line4)
    line5_x = find_first_black(line5)
    line6_x = find_first_black(line6)
    line7_x = find_first_black(line7)
    line8_x = find_first_black(line8)
    line9_x = find_first_black(line9)
    line10_x = find_first_black(line10)
    line11_x = find_first_black(line11)
    line12_x = find_first_black(line12)
    line13_x = find_first_black(line13)
    line14_x = find_first_black(line14)
    line15_x = find_first_black(line15)
    line16_x = find_first_black(line16)

    cv2.circle(frame, (line1_x, 11), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line2_x, 23), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line3_x, 39), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line4_x, 55), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line5_x, 68), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line6_x, 83), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line7_x, 100), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line8_x, 114), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line9_x, 129), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line10_x, 145), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line11_x, 161), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line12_x, 175), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line13_x, 192), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line14_x, 207), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line15_x, 221), 2, (0, 255, 0), -1)
    cv2.circle(frame, (line16_x, 236), 2, (0, 255, 0), -1)

    f.write(str([line1_x, line2_x, line3_x, line4_x, line5_x, line6_x, line7_x, line8_x, line9_x, line10_x, line11_x, line12_x, line13_x, line14_x, line15_x, line16_x]))
    f.write("\n")
    rows, cols = gray.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),-1.5,1)
    rotated = cv2.warpAffine(gray,M,(cols,rows))

    cv2.imshow('image',frame)
    # cv2.imshow('line',line_p)
    time.sleep(0.05)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()