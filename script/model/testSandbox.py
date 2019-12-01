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
import math

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
    if math.isnan(min) or math.isnan(max):
        #weights map
        foreground_weight = np.amin(image)
        defect_weight = np.amax(image)
        foreground_pixel_count = np.count_nonzero(image == foreground_weight)
        defect_pixel_count = np.count_nonzero(image == defect_weight)
        #sum_pixels = foreground_pixel_count + defect_pixel_count
        defect_size_ratio = (float)(defect_pixel_count) / (float)(foreground_pixel_count + defect_pixel_count)
        defect_size_ratio_perc = round(defect_size_ratio * 100.0, 4)
        info = 'foregroundW = ' + str(round(foreground_weight, 4)) + ', defectW = ' + str(round(defect_weight, 4)) + ', Area = ' + str(defect_size_ratio_perc) + '%'
        #add text top left
        image_plot.text(10, 20, info, style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.6, 'pad': 5})

        im = plt.imshow(image, cmap=colorspace)
    else:
        im = plt.imshow(image, cmap=colorspace, vmin=min, vmax=max)
    divider = make_axes_locatable(image_plot)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def make_four_graph_stack(name_1, image_1, colorSpace_1, vmin_1, vmax_1,
                          name_2, image_2, colorSpace_2, vmin_2, vmax_2,
                          name_3, image_3, colorSpace_3, vmin_3, vmax_3,
                          name_4, image_4, colorSpace_4, vmin_4, vmax_4,
                          save_path):
    fig = plt.figure(figsize=(12, 9))
    add_subplot(fig, 2, 2, 1, name_1, image_1, colorSpace_1, vmin_1, vmax_1)
    add_subplot(fig, 2, 2, 2, name_2, image_2, colorSpace_2, vmin_2, vmax_2)
    add_subplot(fig, 2, 2, 3, name_3, image_3, colorSpace_3, vmin_3, vmax_3)
    add_subplot(fig, 2, 2, 4, name_4, image_4, colorSpace_4, vmin_4, vmax_4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    #plt.show()

def get_inner_part_weight(label, edges):
    adjusted_label = label - edges
    adjusted_label[adjusted_label < 0.0] = 0.0
    adjusted_label_tensor = image2tensor(adjusted_label)
    weights_tensor = get_weight_matrix(adjusted_label_tensor)
    weights_numpy = K.eval(weights_tensor)  # basically tensor to numpy
    weightsImage_numpy = weights_numpy[0, :, :, 0]
    return weightsImage_numpy

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
        imageN, labelN = adjustData(image, label, False, 1)

        #calculate weights
        weights_tensor = get_weight_matrix(label_tensorN)
        weights_numpy = K.eval(weights_tensor)#basically tensor to numpy
        weightsImage_numpy = weights_numpy[0,:,:,0]

        emptyImage = np.zeros_like(image)
        emptyWeights = np.ones_like(image)
        base_output_path = 'C:/src/figs/'

        make_four_graph_stack("Image", image, "gray", 0, 255,
                              "Label", label, "gray", 0, 1,
                              "Edges_empty", emptyImage, "gray", 0, 1,
                              "Weights", weightsImage_numpy, "viridis", float('nan'), float('nan'),
                              base_output_path + file_name + '_edges_less_1.png')

        # calculate weights
        edgeImage_30_90_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.9)
        edgeImage_30_90_numpy = edgeImage_30_90_numpy * labelN #leave only edge that is in the label
        innerWeight_30_90_numpy = get_inner_part_weight(labelN, edgeImage_30_90_numpy)
        make_four_graph_stack("Image", image, "gray", 0, 255,
                              "Label", label, "gray", 0, 1,
                              "Edges [3x3 kernel fill < 90%]", edgeImage_30_90_numpy, "gray", 0, 1,
                              "Weights", innerWeight_30_90_numpy, "viridis", float('nan'), float('nan'),
                              base_output_path + file_name + '_edges_less_90.png')

        # calculate weights
        edgeImage_30_80_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.8)
        edgeImage_30_80_numpy = edgeImage_30_80_numpy * labelN  # leave only edge that is in the label
        innerWeight_30_80_numpy = get_inner_part_weight(labelN, edgeImage_30_80_numpy)
        make_four_graph_stack("Image", image, "gray", 0, 255,
                              "Label", label, "gray", 0, 1,
                              "Edges [3x3 kernel fill < 80%]", edgeImage_30_80_numpy, "gray", 0, 1,
                              "Weights", innerWeight_30_80_numpy, "viridis", float('nan'), float('nan'),
                              base_output_path + file_name + '_edges_less_80.png')

        # calculate weights
        edgeImage_30_70_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.7)
        edgeImage_30_70_numpy = edgeImage_30_70_numpy * labelN  # leave only edge that is in the label
        innerWeight_30_70_numpy = get_inner_part_weight(labelN, edgeImage_30_70_numpy)
        make_four_graph_stack("Image", image, "gray", 0, 255,
                              "Label", label, "gray", 0, 1,
                              "Edges [3x3 kernel fill < 70%]", edgeImage_30_70_numpy, "gray", 0, 1,
                              "Weights", innerWeight_30_70_numpy, "viridis", float('nan'), float('nan'),
                              base_output_path + file_name + '_edges_less_70.png')

        # calculate weights
        edgeImage_30_60_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.6)
        edgeImage_30_60_numpy = edgeImage_30_60_numpy * labelN  # leave only edge that is in the label
        innerWeight_30_60_numpy = get_inner_part_weight(labelN, edgeImage_30_60_numpy)
        make_four_graph_stack("Image", image, "gray", 0, 255,
                              "Label", label, "gray", 0, 1,
                              "Edges [3x3 kernel fill < 60%]", edgeImage_30_60_numpy, "gray", 0, 1,
                              "Weights", innerWeight_30_60_numpy, "viridis", float('nan'), float('nan'),
                              base_output_path + file_name + '_edges_less_60.png')

        # calculate weights
        edgeImage_30_50_numpy = getEdgeMatrix(label_tensorN, 0.3, 0.5)
        edgeImage_30_50_numpy = edgeImage_30_50_numpy * labelN  # leave only edge that is in the label
        innerWeight_30_50_numpy = get_inner_part_weight(labelN, edgeImage_30_50_numpy)
        make_four_graph_stack("Image", image, "gray", 0, 255,
                              "Label", label, "gray", 0, 1,
                              "Edges [3x3 kernel fill < 50%]", edgeImage_30_50_numpy, "gray", 0, 1,
                              "Weights", innerWeight_30_50_numpy, "viridis", float('nan'), float('nan'),
                              base_output_path + file_name + '_edges_less_50.png')


        """
        add_subplot(fig, 2, 4, 1, "Image", image, "gray", 0, 255)
        add_subplot(fig, 2, 4, 2, "Label", label, "gray", 0, 1)
        add_subplot(fig, 2, 4, 3, "Weights", weightsImage_numpy, "viridis", 0, 3)
        add_subplot(fig, 2, 4, 4, "Edge >50% && <80%", edgeImage_50_80_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 5, "Edge >30% && <80%", edgeImage_30_80_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 6, "Edge >10% && <80%", edgeImage_10_80_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 7, "Edge >30% && <60%", edgeImage_30_60_numpy, "gray", 0, 1)
        add_subplot(fig, 2, 4, 8, "Edge >10% && <50%", edgeImage_10_50_numpy, "gray", 0, 1)
        """

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

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()