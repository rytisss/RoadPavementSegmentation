import cv2
import numpy as np

class Filtering:
    @staticmethod
    def MinArea(image, minArea):
        if minArea > 0.0:
            contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            filteredOutContours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= minArea:
                    filteredOutContours.append(contour)
            #draw black blob on filter out contours
            filteredImage = np.copy(image)
            cv2.drawContours(filteredImage, filteredOutContours, -1, 0, -1)
            return filteredImage
        else:
            filteredImage = np.copy(image)
            return filteredImage