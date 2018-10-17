import cv2
import numpy as np

img = cv2.imread('yarn.jpeg',0) #import image
equ = cv2.equalizeHist(img) #equalize using opencv
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res) #save the image
cv2.imshow('res.png',res) #show the image
cv2.waitKey(0)
cv2.destroyAllWindows()