import cv2

cap1 = cv2.VideoCapture('out1.avi')



while True:
	ret1, img1 = cap1.read()

	if ret1:
	  cv2.imshow('img1',img1)

	  k = cv2.waitKey(100) 
	  if k == 27: #press Esc to exit
	     break


cap1.release()

cv2.destroyAllWindows()

