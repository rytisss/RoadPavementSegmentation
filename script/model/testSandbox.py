from script.model.utilities import *
from script.model.losses import *
import numpy as np
import keras.backend as K
import cv2
import glob
import tensorflow as tf

def image2tensor(image):
    image_shape = image.shape
    print('image shape:\n')
    print(image_shape)
    tensor = image.reshape(1, image_shape[0], image_shape[1], 1)
    tensor_shape = tensor.shape
    print('tensor shape:\n')
    print(tensor_shape)
    return tensor

def tensor2image(tensor):
    image = tensor[0,:,:,:]
    return image

def visualize_tensor(tensor, windowName):
    image = tensor2image(tensor)
    image *= 255
    cv2.imshow(windowName, image)
    cv2.waitKey(1)

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
        #make tensor
        label_tensor = image2tensor(label)
        image_tensor = image2tensor(image)
        image_tensorN, label_tensorN = adjustData(image_tensor, label_tensor, False, 1)
        #calculate weights
        weights_tensor = get_weight_matrix(label_tensorN)
        print(type(weights_tensor))
        weights_numpy = K.eval(weights_tensor)#basically tensor to numpy

        #find edge
        edge_tensor = get_edge_matrix(label_tensorN)
        print(type(edge_tensor))
        edge_numpy = K.eval(edge_tensor)  # basically tensor to numpy
        visualize_tensor(edge_numpy, "edge")
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()