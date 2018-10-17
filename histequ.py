import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('yarn.jpeg',0) #import image
equ = cv2.equalizeHist(img) #equalize using opencv
cv2.imwrite('equalized.png',equ)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res) #save the image
cv2.imshow('res.png',res) #show the image

hist,bins = np.histogram(img.flatten(),256,[0,256]) #creates the bins for the histogram
hist2,bins = np.histogram(img.flatten(),256,[0,256]) #creates the bins for the histogram
cdf = hist.cumsum() #does cummulative summation of the histogram
cdf_normalized = cdf * hist.max() / cdf.max() #normalizes cdf

plt.figure(1)
plt.subplot(211)
plt.plot(cdf_normalized, color = 'b') #creates the plot
plt.hist(img.flatten(),256,[0,256], color = 'r') #plots the histogram in color red
plt.xlim([0,256]) #set the x limits
plt.legend(('cdf','histogram'), loc = 'upper left') #draws the legends associated with the axes

img2 = cv2.imread('equalized.png',0)

plt.subplot(212)
plt.plot(cdf_normalized, color = 'b') #creates the plot
plt.hist(img2.flatten(),256,[0,256], color = 'r') #plots the histogram in color red
plt.xlim([0,256]) #set the x limits
plt.legend(('cdf','histogram'), loc = 'upper left') #draws the legends associated with the axes
plt.show() #shows the plot

cv2.waitKey(0)
cv2.destroyAllWindows()