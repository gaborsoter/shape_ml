import numpy as np
import cv2
import time

f= open("image_processing_dots_lines.txt","w+")

cap = cv2.VideoCapture('data/08_14/lines/batch_1000_1.avi')
cv2.namedWindow('image')

def nothing(x):
    pass

def sorting(centres):

    return output_centres

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 2

cv2.createTrackbar('thresh1','image',3,1001,nothing)
cv2.createTrackbar('erosion','image',0,101,nothing)
iii = 0
while(True):
    print(iii)
    iii += 1
    thresh1  = cv2.getTrackbarPos('thresh1','image')
    thresh2  = cv2.getTrackbarPos('erosion','image')

    if thresh1 % 2 == 0:
        thresh1 += 1

    
    ret, frame = cap.read()

    # rotation
    rows, cols, ch = frame.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),2,1)
    frame = cv2.warpAffine(frame,M,(cols,rows))

    # roi
    frame = frame[60:400, 148:485]
    # removing noise
    blur = cv2.GaussianBlur(frame, (int(3), int(3)), 0)
    # grayscaling
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # thresholding
    ret,thresh = cv2.threshold(gray,121,255,cv2.THRESH_BINARY)
    
    # cleaning with erosion
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 2)
    dilation = cv2.dilate(erosion,kernel,iterations = 6)

    im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[6]
    centres = []
    np_centres = []
    for i in range(len(contours)):
        moments = cv2.moments(contours[i])
        if (moments['m00'] == 0 or moments['m00'] == 0):
            mx = 0
        else:
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
            np_centres.append(np.array((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))))

    print('centres: ', len(centres))
    np_centres = np.array(np_centres)


    sort_row = np_centres[np_centres[:, 0].argsort()]
    # print('sort_row: ', sort_row)
    sort_all = np.empty((10,2), int)
    delete_index = [i for i in range(10)]
    for i in range(10):
        temp = sort_row[i*10:i*10+10]
        sort_column = temp[temp[:, 1].argsort()]
        sort_all = np.concatenate((sort_all, sort_column))

    sort_all = np.delete(sort_all, delete_index, axis = 0)

    for i in range(sort_all.shape[0]):
        cv2.circle(frame, (sort_all[i][0], sort_all[i][1]), 6, (0, 255, 0), -1)
        cv2.putText(frame,str(i), (sort_all[i][0], sort_all[i][1] + 2), font, fontScale, fontColor, lineType)

    f.write(str(sort_all.tolist())+"\n")

    print('sort_all: ', sort_all.shape[0])
    if sort_all.shape[0] != 100:
        print('NOOOOOOOOOOOOOOOOOOOOOO')
        time.sleep(0.05)
    cv2.imshow('image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

f.close() 