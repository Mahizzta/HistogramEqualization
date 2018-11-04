import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('dog.jpg',0)
src = cv2.imread('dog.jpg',0)

a = np.zeros((256,),dtype=np.float16)
b = np.zeros((256,),dtype=np.float16)

height,width=img.shape

# Find original histogram
for i in range(width):
    for j in range(height):
        g = img[j,i]
        a[g] = a[g]+1

print(a,"\nThis is a")

# Do histogram equalization

tmp = 1.0/(height*width)
b = np.zeros((256,),dtype=np.float16)
for i in range(256):
    for j in range(i+1):
        b[i] += a[j] * tmp
    b[i] = round(b[i] * 255)

b=b.astype(np.uint8)

print(b,"\nThis is b")

# Take values from equalized histogram and turn into new image
for i in range(width):
    for j in range(height):
        g = img[j,i]
        img[j,i]= b[g]

# Save, stack, and show before and after
cv2.imwrite('equalized.png',img)
res = np.hstack((src,img)) #stacking images side-by-side
cv2.imshow('Before VS After',res)
cv2.imwrite('res.png',res)
print("Here is a before and after picture of the histogram equalization")
print("A plot of the histogram before and after is also show")
print("The output image has been saved as equalized.png in root folder")

hist,bins = np.histogram(src.flatten(),256,[0,256]) #creates the bins for the histogram
cdf = hist.cumsum() #does cummulative summation of the histogram
cdf_normalized = cdf * hist.max() / cdf.max() #normalizes cdf

# Plot original histogram
plt.figure(1)
plt.subplot(211)
plt.plot(cdf_normalized, color = 'b') #creates the plot
plt.hist(src.flatten(),256,[0,256], color = 'r') #plots the histogram in color red
plt.xlim([0,256]) #set the x limits
plt.legend(('cdf','histogram'), loc = 'upper left') #draws the legends associated with the axes

hist2,bins2 = np.histogram(img.flatten(),256,[0,256]) #creates the bins for the histogram
cdf2 = hist2.cumsum() #does cummulative summation of the histogram
cdf2_normalized = cdf2 * hist2.max() / cdf2.max() #normalizes cdf

# Plot equalized histogram
plt.subplot(212)
plt.plot(cdf2_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
