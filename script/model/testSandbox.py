from script.model.utilities import *
from script.model.losses import *
import numpy as np
import keras.backend as K
import cv2
import glob
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plotWeightsMatrix(image, name, path):
    fig = plt.figure(figsize=(8, 8))
    plt.title.set_text(name)
    plt.imshow(image, cmap='viridis')
    plt.colorbar()
    plt.show()

def plotGrayscaleMatrix(image, name, path):
    fig = plt.figure(figsize=(8, 8))
    plt.title.set_text(name)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

def add_subplot(fig, rows, cols, pos, name, image, colorspace, min, max):
    image_plot = fig.add_subplot(rows, cols, pos)
    image_plot.title.set_text(name)
    im = plt.imshow(image, cmap=colorspace, vmin=min, vmax=max)
    divider = make_axes_locatable(image_plot)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def main():
    # image path
    data_input = "C:/Users/Rytis/Desktop/Set_0/Train/"
    images_path = data_input + 'Images/'
    labels_path = data_input + 'Labels/'
    # gather all data in image and label directories
    images = glob.glob(images_path + '*.bmp')
    labels = glob.glob(labels_path + '*.bmp')
    # open and process
    for i in range(0, len(images)):
        image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        file_name = os.path.splitext(os.path.basename(images[i]))[0]
        #make tensor
        label_tensor = image2tensor(label)
        image_tensor = image2tensor(image)
        image_tensorN, label_tensorN = adjustData(image_tensor, label_tensor, False, 1)
        #calculate weights
        weights_tensor = get_weight_matrix(label_tensorN)
        print(type(weights_tensor))
        weights_numpy = K.eval(weights_tensor)#basically tensor to numpy
        weightsImage_numpy = weights_numpy[0,:,:,0]

        edgeImage_50_80_numpy = getEdgeMatrix(label_tensorN, 0.5, 0.8)
        edgeImage_30_80_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.8)
        edgeImage_10_80_numpy = getEdgeMatrix(label_tensorN, 0.1, 0.8)
        edgeImage_30_60_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.6)
        edgeImage_10_50_numpy = getEdgeMatrix(label_tensorN, 0.1, 0.5)


        fig = plt.figure(figsize=(8, 8))

        add_subplot(fig, 2, 4, 1, "Image", image, "gray", 0, 255)
        add_subplot(fig, 2, 4, 2, "Label", label, "gray", 0, 1)
        add_subplot(fig, 2, 4, 3, "Weights", weightsImage_numpy, "viridis", 0, 3)
        add_subplot(fig, 2, 4, 4, "Edge >50% && <80%", edgeImage_50_80_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 5, "Edge >30% && <80%", edgeImage_30_80_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 6, "Edge >10% && <80%", edgeImage_10_80_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 7, "Edge >30% && <60%", edgeImage_30_60_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 8, "Edge >10% && <50%", edgeImage_10_50_numpy, "gray", 0, 1)

        """
        image_plot = fig.add_subplot(2, 4, 1)
        image_plot.title.set_text('Image')
        im = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        divider = make_axes_locatable(image_plot)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        label_plot = fig.add_subplot(2, 4, 2)
        label_plot.title.set_text('Label')
        plt.imshow(label, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()

        weight_plot = fig.add_subplot(2, 4, 3)
        weight_plot.title.set_text('Weights')
        plt.imshow(weightsImage_numpy, cmap='viridis')
        plt.colorbar()

        edge0508_plot = fig.add_subplot(2, 4, 4)
        edge0508_plot.title.set_text('Edge >50% && <80%')
        plt.imshow(edgeImage_50_80_numpy, cmap='gray')
        plt.colorbar()

        edge0308_plot = fig.add_subplot(2, 4, 5)
        edge0308_plot.title.set_text('Edge >30% && <80%')
        plt.imshow(edgeImage_30_80_numpy, cmap='gray')
        plt.colorbar()

        edge0108_plot = fig.add_subplot(2, 4, 6)
        edge0108_plot.title.set_text('Edge >10% && <80%')
        plt.imshow(edgeImage_10_80_numpy, cmap='gray')
        plt.colorbar()

        edge0306_plot = fig.add_subplot(2, 4, 7)
        edge0306_plot.title.set_text('Edge >30% && <60%')
        plt.imshow(edgeImage_30_60_numpy, cmap='gray')
        plt.colorbar()

        edge0105_plot = fig.add_subplot(2, 4, 8)
        edge0105_plot.title.set_text('Edge >10% && <50%')
        plt.imshow(edgeImage_10_50_numpy, cmap='gray')
        plt.colorbar()

        """

        plt.show()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()