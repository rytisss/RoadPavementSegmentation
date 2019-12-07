import os
import cv2
import numpy as np

#################################################
imageDir = 'C:/Users/Rytis/Desktop/Set_0/Train/Augm/Labels/035_rot0.bmp'
image = cv2.imread(imageDir, cv2.IMREAD_GRAYSCALE)
width, height = image.shape
#draw border
borderColor = 0

#invert image
image = abs(255 - image)
cv2.rectangle(image, (0,0), (height - 1, width - 1), borderColor, 1)
cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/003_rot180_label.bmp', image)
cv2.imshow("image", image)
cv2.waitKey(0)
