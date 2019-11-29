from script.model.utilities import *
from script.model.losses import *
import numpy as np
import keras.backend as K
import cv2
import glob
import matplotlib.pyplot as plt
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

def image2threeDim(image):
    image_shape = image.shape
    print('image shape:\n')
    print(image_shape)
    image3dim = image.reshape(image_shape[0], image_shape[1], 1)
    return image3dim

def tensor2image(tensor):
    image = tensor[0,:,:,:]
    return image

def visualize_tensor(tensor, windowName):
    image = tensor2image(tensor)
    image *= 255
    cv2.imshow(windowName, image)
    cv2.waitKey(1)

def getEdgeMatrix(label_tensorN, min_overlay = 0.5, max_overlay = 0.8):
    edge_tensor = get_edge_matrix(label_tensorN, min_overlay, max_overlay)
    print(type(edge_tensor))
    edge_numpy = K.eval(edge_tensor)  # basically tensor to numpy
    edgeImage_numpy = edge_numpy[0, :, :, 0]
    return edgeImage_numpy

def main():
    # image path
    data_input = "E:/RoadCracksInspection/datasets/Set_0/Train/"
    images_path = data_input + 'Images/'
    labels_path = data_input + 'Labels/'
    # gather all data in image and label directories
    images = glob.glob(images_path + '*.bmp')
    labels = glob.glob(labels_path + '*.bmp')
    # open and process
    for i in range(0, len(images)):
        image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        #make tensor
        label_tensor = image2tensor(label)
        image_tensor = image2tensor(image)
        image_tensorN, label_tensorN = adjustData(image_tensor, label_tensor, False, 1)
        #calculate weights
        weights_tensor = get_weight_matrix(label_tensorN)
        print(type(weights_tensor))
        weights_numpy = K.eval(weights_tensor)#basically tensor to numpy
        weightsImage_numpy = weights_numpy[0,:,:,0]

        edgeImage_50_80_numpy = getEdgeMatrix(label_tensorN)

        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 2, 1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        fig.add_subplot(2, 2, 2)
        plt.imshow(label, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        fig.add_subplot(2, 2, 3)
        plt.imshow(weightsImage_numpy, cmap='viridis')
        plt.colorbar()
        fig.add_subplot(2, 2, 4)
        plt.imshow(edgeImage_50_80_numpy, cmap='gray')
        plt.colorbar()

        plt.show()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()