import os
import cv2
import numpy as np
import glob
from script.processing.DataAnalysis.render import Render

#global variable
is_clicked = False
first_pt = (0,0)
second_pt = (0,0)
image = np.zeros((1,1), dtype=np.int8)
image_with_render = np.zeros((1,1), dtype=np.int8)
label = np.zeros((1,1), dtype=np.int8)
preview = np.zeros((1,1), dtype=np.int8)
preview_window = 'preview'

def mouse_event(event, x, y, flags, params):
    global is_clicked
    global first_pt
    global second_pt
    global image
    global image_with_render
    global label
    global preview_window
    global preview
    current_pos = (x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_clicked == False:
            print('Down: ', x, ' ', y)
            is_clicked = True
            first_pt = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        is_clicked = False
        print('Up: ', x, ' ', y)
        second_pt = (x,y)
    if is_clicked:
        #preview window
        preview = np.copy(image)
        cv2.rectangle(preview, first_pt, current_pos, 0, 1)
        cv2.imshow(preview_window, preview)
        cv2.waitKey(1)

#################################################
#gather all images

imagesPath = 'C:/Users\Rytis/Desktop/Set_0/Train/shuffled/AUGM/Images/'
labelsPath = 'C:/Users\Rytis/Desktop/Set_0/Train/shuffled/AUGM/Labels/'
images = glob.glob(imagesPath + '*.bmp')
labels = glob.glob(labelsPath + '*.bmp')

cv2.namedWindow(preview_window)
cv2.setMouseCallback(preview_window, mouse_event)
for i in range(0, len(imagesPath)):
    image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    preview = np.copy(image)
    width, height = image.shape
    label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
    #invert
    label = abs(255-label)
    #get name of image

    # render defect on image
    outerColor = (0, 0, 200)
    innerColor = (0, 0, 80)
    #renderedImage = Render.Defects(image, prediction, outerColor, innerColor)
    #draw border
    #borderColor = 0

    #invert image
    #cv2.rectangle(image, (0,0), (height - 1, width - 1), borderColor, 1)
    #cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/23_116_rot180.bmp', image)
    cv2.imshow("image", image)
    cv2.imshow("label", label)
    cv2.imshow(preview_window, preview)
    key = cv2.waitKey(0)

    if key == 's':
        outputDir = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS'
        cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/23_116_rot180.bmp', image)
