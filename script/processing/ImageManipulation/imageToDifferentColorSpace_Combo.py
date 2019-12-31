import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_label(label):
    first_addition = 'Prediction of model trained with '
    second_addition = ' function'
    if '_predict_ce25dice75' in label:
        return first_addition + r'$\mathit{L}_{\mathit{CE}}^{\mathbf{25\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{75\%}}$' + second_addition
    if '_predict_ce50dice50' in label:
        return first_addition + r'$\mathit{L}_{\mathit{CE}}^{\mathbf{50\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{50\%}}$' + second_addition
    if '_predict_ce75dice25' in label:
        return first_addition + r'$\mathit{L}_{\mathit{CE}}^{\mathbf{75\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{25\%}}$' + second_addition
    if '_predict_surfaceNdice' in label:
        return first_addition + r'$\mathit{L}_{\mathit{DB}}$' + second_addition
    if '_wce25dice75' in label:
        return first_addition + r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{25\%}}$'+'+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{75\%}}$' + second_addition
    if '_wce50dice50' in label:
        return first_addition + r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{50\%}}$'+'+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{50\%}}$' + second_addition
    if '_predict_wce75dice25' in label:
        return first_addition + r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{75\%}}$'+'+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{25\%}}$' + second_addition
    return label

def get_file_name_only(path):
    file_name_with_ext = path.rsplit('\\', 1)[1]
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    return file_name

def get_image_width_n_height(image):
    width = image.shape[0]
    height = image.shape[1]
    return width, height

def draw_frame(image):
    width, height = get_image_width_n_height(image)
    borderColor = 0
    cv2.rectangle(image, (0, 0), (height - 1, width - 1), borderColor, 1)
    return image

def threshold_n_invert_n_draw_frame(image):
    _, image_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # invert image
    image_th = abs(255 - image_th)
    image_th = draw_frame(image_th)
    return image_th

def add_subplot(fig, rows, cols, pos, name, image, colorspace, min, max):
    image_plot = fig.add_subplot(rows, cols, pos)
    name = parse_label(name)
    image_plot.title.set_text(name)
    im = plt.imshow(image, cmap=colorspace, vmin=min, vmax=max)
    divider = make_axes_locatable(image_plot)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def make_single_graph(name, image, save_path):
    #threshold
    _, image_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # invert image
    image_th = abs(255 - image_th)
    fig = plt.figure(figsize=(6.6, 4.8))
    norm_image = image / 255.
    colormap = 'viridis'
    vmin = 0.0
    vmax = 1.0
    add_subplot(fig, 1, 1, 1, name, norm_image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def make_single_graph_grayscale(name, image, save_path):
    fig = plt.figure(figsize=(6.6, 4.8))
    norm_image = image / 255.
    colormap = 'gray_r'
    vmin = 0.0
    vmax = 1.0
    add_subplot(fig, 1, 1, 1, name, norm_image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def make_single_graph_full(name, image, save_path):
    fig = plt.figure(figsize=(6.6, 4.8))
    colormap = 'gray'
    vmin = 0.0
    vmax = 255.0
    add_subplot(fig, 1, 1, 1, name, image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

#################################################
label_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/insight comparisson/combo/label/'
image_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/insight comparisson/combo/image/'
prediction_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/insight comparisson/combo/prediction/'
image_name_segments = ['005','16', '27', '28', '34']
output = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/insight comparisson/combo/'

#gather all images from directory
image_paths = glob.glob(image_path + '*.bmp')
label_paths = glob.glob(label_path + '*.bmp')
prediction_paths = glob.glob(prediction_path + '*.bmp')
for image_path in image_paths:
    # get label image name
    image_name = get_file_name_only(image_path)
    for image_name_segment in image_name_segments:
        if image_name_segment in image_name:
            # Open image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Draw frame
            image = draw_frame(image)
            cv2.imshow('image', image)
            cv2.waitKey(1)
            # Save image
            cv2.imwrite(output + image_name + '_.bmp', image)
            make_single_graph_full('Image', image, output + image_name + 'graph_.png')
            # Search right label image
            for label_path in label_paths:
                label_name = get_file_name_only(label_path)
                if image_name_segment in label_name:
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    make_single_graph_grayscale('Label', label, output + label_name + 'graph_.png')
                    make_single_graph('Label', label, output + label_name + 'color_graph_.png')
                    label = threshold_n_invert_n_draw_frame(label)
                    cv2.imshow('label', label)
                    cv2.waitKey(1)
                    # Save image
                    cv2.imwrite(output + label_name + '_.bmp', label)
            # Search right prediction image
            for prediction_path in prediction_paths:
                prediction_name = get_file_name_only(prediction_path)
                if image_name_segment in prediction_name:
                    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
                    make_single_graph(prediction_name, prediction, output + prediction_name + 'graph_.png')
                    # threshold at 50%
                    _, prediction_th = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
                    make_single_graph(prediction_name, prediction_th, output + prediction_name + 'graph50_.png')
                    prediction = threshold_n_invert_n_draw_frame(prediction)
                    cv2.imshow('prediction', prediction)
                    cv2.waitKey(1)
                    # Save image
                    cv2.imwrite(output + prediction_name + '_.bmp', prediction)

