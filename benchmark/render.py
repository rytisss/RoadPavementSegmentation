import cv2
import numpy as np

class Render:
    @staticmethod
    def Defects(image, predictionImage, outerColor, transparentColor):
        #find contours
        contours, _ = cv2.findContours(predictionImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        imageHeight, imageWidth  = image.shape
        #Draw filled contours on dummy and add it with original image in RGB and then draw outer contours
        transparentMask = np.zeros((imageHeight,imageWidth,3), np.uint8)
        cv2.drawContours(transparentMask, contours, -1, transparentColor, -1)
        imageRGB =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        outputImage = cv2.add(imageRGB, transparentMask)
        #cv2.drawContours(outputImage, contours, -1, outerColor, 1)
        return outputImage