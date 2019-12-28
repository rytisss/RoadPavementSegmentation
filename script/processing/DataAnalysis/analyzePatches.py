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
output_dir = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/'
image_name = ''
#store all regions coordinates that belongs to same image
rois = []

def render_border(image):
    width = image.shape[0]
    height = image.shape[1]
    cv2.rectangle(image, (0,0), (height - 1, width - 1), 0, 1)
    return image

def get_centered_rectangle_p(left, top, right, bottom):
    roi = [left, top, right, bottom]
    return get_centered_rectangle(roi)

def get_centered_rectangle(rectangle):
    width = 50
    height = 50
    left = rectangle[0]
    top = rectangle[1]
    right = rectangle[2]
    bottom = rectangle[3]
    center_x = ((left + right) / 2)
    center_y = ((top + bottom) / 2)
    centered_rect_left = (int)(center_x - width / 2)
    centered_rect_right = (int)(center_x + width / 2)
    centered_rect_top = (int)(center_y - height / 2)
    centered_rect_bottom = (int)(center_y + height / 2)
    centered_rect = []
    centered_rect.append(centered_rect_left)
    centered_rect.append(centered_rect_top)
    centered_rect.append(centered_rect_right)
    centered_rect.append(centered_rect_bottom)
    return centered_rect

def render_regions(image, rois):
    for i in range(0, (int)(len(rois) / 4)):
        left = rois[i * 4 + 0]
        top = rois[i * 4 + 1]
        right = rois[i * 4 + 2]
        bottom = rois[i * 4 + 3]
        cv2.rectangle(image, (top, left), (bottom, right), 0, 1)
    return image

def save_crop(path, left, top, right, bottom, image, label, image_crop, label_crop):
    global rois
    rois.append(left)
    rois.append(top)
    rois.append(right)
    rois.append(bottom)

    outer_color = (0, 0, 200)
    inner_color = (0, 0, 70)

    #invert label
    inverted_label = abs(255 - label)
    rendered_image = Render.Defects(image, inverted_label, outer_color, inner_color)
    rendered_image = render_regions(rendered_image, rois)
    rendered_image = render_border(rendered_image)
    save_preview_window = 'image_rois'
    cv2.imshow(save_preview_window, rendered_image)

    #save cropped image and labels, save image 2 times, showing good and bad regions
    #bad region
    #invert cropped label
    inverted_label_crop = abs(255 - label_crop)
    rendered_crop_label_bad = Render.Defects(image_crop, inverted_label_crop, outer_color, inner_color)
    rendered_crop_label_bad = render_border(rendered_crop_label_bad)
    #good region
    outer_color = (0, 200, 0)
    inner_color = (0, 40, 0)
    rendered_crop_label_good = Render.Defects(image_crop, label_crop, outer_color, inner_color)
    rendered_crop_label_good = render_border(rendered_crop_label_good)
    #render border to image crop and label crop
    label_crop = render_border(label_crop)
    image_crop = render_border(image_crop)

    crop_name = '_' + str(left) + '_' + str(top) + '_' + str(right) + '_' + str(bottom)

    print('saving...')
    image = render_border(image)
    image = render_regions(image, rois)
    cv2.imwrite(path + 'image.bmp', image)
    label = render_border(label)
    cv2.imwrite(path + 'label.bmp', label)
    cv2.imwrite(path + 'rendered_image.bmp', rendered_image)
    #save crops
    cv2.imwrite(path + crop_name + '_label.bmp', label_crop)
    cv2.imwrite(path + crop_name + '_image.bmp', image_crop)
    cv2.imwrite(path + crop_name + '_defect.bmp', rendered_crop_label_bad)
    cv2.imwrite(path + crop_name + '_good.bmp', rendered_crop_label_good)

    cv2.waitKey(3000)
    cv2.destroyWindow(save_preview_window)

def mouse_event(event, x, y, flags, params):
    global is_clicked
    global first_pt
    global second_pt
    global image
    global image_with_render
    global label
    global preview_window
    global preview
    global output_dir
    global image_name
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
        #clip region from label and image and maximize them
        crop_image_window = 'crop_image'
        crop_label_window = 'crop_label'
        crop_render_window = 'crop_render'
        width, height = image.shape
        resize_ratio = 5
        x0 = x1 = y0 = y1 = 0
        if first_pt[1] > second_pt[1]:
            x0 = second_pt[1]
            x1 = first_pt[1]
        else:
            x0 = first_pt[1]
            x1 = second_pt[1]

        if first_pt[0] > second_pt[0]:
            y0 = second_pt[0]
            y1 = first_pt[0]
        else:
            y0 = first_pt[0]
            y1 = second_pt[0]

        center_roi = get_centered_rectangle_p(x0, y0, x1, y1)
        #assign centered roi
        x0 = center_roi[0]
        y0 = center_roi[1]
        x1 = center_roi[2]
        y1 = center_roi[3]

        crop_width = x1 - x0
        crop_height = y1 - y0

        crop_image = image[x0:x1, y0:y1]
        crop_image = cv2.resize(crop_image, (crop_height * resize_ratio, crop_width * resize_ratio), interpolation=cv2.INTER_NEAREST)
        crop_label = label[x0:x1, y0:y1]
        crop_label = cv2.resize(crop_label, (crop_height * resize_ratio, crop_width * resize_ratio), interpolation=cv2.INTER_NEAREST)
        cv2.namedWindow(crop_image_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(crop_label_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(crop_image_window, crop_image)
        cv2.imshow(crop_label_window, crop_label)
        key = cv2.waitKey(0)
        cv2.destroyWindow(crop_image_window)
        cv2.destroyWindow(crop_label_window)
        if key == 115:#'s'
            #saving
            formed_output_dir = output_dir + image_name + '/'
            if not os.path.exists(formed_output_dir):
                os.makedirs(formed_output_dir)
            print('Saving in: ' + formed_output_dir)

            save_crop(formed_output_dir, x0, y0, x1, y1, image, label, crop_image, crop_label)

            #cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/23_116_rot180.bmp', image)

    if is_clicked:
        #preview window
        preview = np.copy(image)
        cv2.rectangle(preview, first_pt, current_pos, 0, 1)
        #additionally draw centered roi
        centered_roi = get_centered_rectangle_p(first_pt[0], first_pt[1], current_pos[0], current_pos[1])
        cv2.rectangle(preview, (centered_roi[0], centered_roi[1]), (centered_roi[2], centered_roi[3]), 50, 1)
        cv2.imshow(preview_window, preview)
        cv2.waitKey(1)

#################################################
#gather all images

imagesPath = 'C:/Users\Rytis/Desktop/Set_0/Images/'
labelsPath = 'C:/Users\Rytis/Desktop/Set_0/Labels/'
images = glob.glob(imagesPath + '*.bmp')
labels = glob.glob(labelsPath + '*.bmp')

cv2.namedWindow(preview_window)
cv2.setMouseCallback(preview_window, mouse_event)
for i in range(0, len(images)):
    image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
    #image_with_render = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    preview = np.copy(image)
    width, height = image.shape
    label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)

    image_name_with_ext = images[i].rsplit('\\', 1)[1]
    image_name, image_ext = os.path.splitext(image_name_with_ext)

    outer_color= (0,0,200)
    inner_color=(0,0,80)
    image_with_render = Render.Defects(image, label, outer_color,inner_color)
    # invert label
    label = abs(255 - label)
    #draw border
    #borderColor = 0

    #invert image
    #cv2.rectangle(image, (0,0), (height - 1, width - 1), borderColor, 1)
    #cv2.imwrite('C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/23_116_rot180.bmp', image)
    cv2.imshow("image", image)
    cv2.imshow("label", label)
    cv2.imshow('render', image_with_render)
    cv2.imshow(preview_window, preview)
    key = cv2.waitKey(0)
    print('Clearing rois...')
    rois.clear()
