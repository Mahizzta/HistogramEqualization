import cv2
import numpy as np

img = cv2.imread('yarn.jpeg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
cv2.imshow('res.png',res)
cv2.waitKey(0)

cv2.destroyAllWindows()