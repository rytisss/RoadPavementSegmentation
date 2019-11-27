from script.model.utilities import *
import numpy as np
import cv2
import glob

def main():
    # image path
    data_input = "D:/RoadCracksInspection/datasets/Set_0/Train/"
    images_path = data_input + 'Images/'
    labels_path = data_input + 'Labels/'
    # gather all data in image and label directories
    images = glob.glob(images_path + '*.bmp')
    labels = glob.glob(labels_path + '*.bmp')
    # open and process
    for i in range(0, len(images)):
        image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        cv2.imshow("image", image)
        cv2.imshow("label", label)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()