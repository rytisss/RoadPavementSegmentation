import os
import cv2
import numpy as np

#################################################
imageDir = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/024.bmp'
image = cv2.imread(imageDir, cv2.IMREAD_GRAYSCALE)
width, height = image.shape
#draw border
borderColor = 0

#invert image
cv2.rectangle(image, (0,0), (height - 1, width - 1), borderColor, 1)
cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/024_border.bmp', image)
cv2.imshow("image", image)
cv2.waitKey(0)