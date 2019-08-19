import cv2

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH,640);
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,640);
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,480);
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

out1 = cv2.VideoWriter('out1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640,480))
out2 = cv2.VideoWriter('out2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640,480))

while True:
	ret1, img1 = cap1.read()
	ret2, img2 = cap2.read()

	out1.write(img1)
	out2.write(img2)

	if ret1 and ret2:
	  cv2.imshow('img1',img1)
	  cv2.imshow('img2',img2) 

	  k = cv2.waitKey(100) 
	  if k == 27: #press Esc to exit
	     break


cap1.release()
cap2.release()

out1.release()
out2.release()


cv2.destroyAllWindows()

