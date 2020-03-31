import os
import cv2
import numpy as np

#################################################
imageDir = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/024_label.bmp'
image = cv2.imread(imageDir, cv2.IMREAD_GRAYSCALE)
width, height = image.shape
#draw border
borderColor = 0

_, image_th = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

#invert image
image = abs(255 - image)
cv2.rectangle(image, (0,0), (height - 1, width - 1), borderColor, 1)
cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/024_label_.bmp', image)
cv2.imshow("image", image)

image_th = abs(255 - image_th)
cv2.rectangle(image_th, (0,0), (height - 1, width - 1), borderColor, 1)
cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/SingleLossFuntionComparisson/021_label_50.bmp', image_th)
cv2.imshow("image50", image_th)
cv2.waitKey(100)

