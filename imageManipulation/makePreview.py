import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_label(label):
    first_addition = 'Prediction of model trained with '
    second_addition = ' function'
    if '_predict_ce' in label:
        return first_addition + r'$\mathit{L}_{\mathit{CE}}$' + second_addition
    if '_predict_dice' in label:
        return first_addition + r'$\mathit{L}_{\mathit{D}}$'+ second_addition
    if '_predict_w60ce' in label:
        return first_addition + r'$\mathit{L}_{\mathit{W60CE}}$'+ second_addition
    if '_predict_w70ce' in label:
        return first_addition + r'$\mathit{L}_{\mathit{W70CE}}$'+ second_addition
    if '_predict_wce' in label:
        return first_addition + r'$\mathit{L}_{\mathit{WCE}}$'+ second_addition
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
image_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500.jpg'
label_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_label.jpg'
baseline = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_unet.jpg'
better_than_baseline = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_unetResWF.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
baseline_prediction = cv2.imread(baseline, cv2.IMREAD_GRAYSCALE)
better_than_baseline_prediction = cv2.imread(better_than_baseline, cv2.IMREAD_GRAYSCALE)

width = int(image.shape[1])
height = int(image.shape[0])
dim = (width, height)
#image = cv2.resize(image, dim)
#label = cv2.resize(label, dim)
baseline_prediction = cv2.resize(baseline_prediction, dim)
better_than_baseline_prediction = cv2.resize(better_than_baseline_prediction, dim)

cv2.imshow('label', label)
cv2.waitKey(1)

output_image_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_.png'
output_label_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_label_.png'
baseline_output_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_unet_.png'
better_than_baseline_output_path = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/result images/20160222_164000_crack500_unetResWF_.png'

make_single_graph_full('Image', image, output_image_path)
make_single_graph_grayscale('Label', label, output_label_path)
make_single_graph_grayscale('UNet Prediction', baseline_prediction, baseline_output_path)
make_single_graph_grayscale('ResUNet+ASPP_WF Prediction', better_than_baseline_prediction, better_than_baseline_output_path)


