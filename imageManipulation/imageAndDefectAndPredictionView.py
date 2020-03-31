import os
import cv2
import numpy as np
import glob
from script.processing.DataAnalysis.render import Render

#global variable
from script.processing.DataAnalysis.statistics import Statistics

is_clicked = False
first_pt = (0,0)
second_pt = (0,0)
image = np.zeros((1,1), dtype=np.int8)
image_with_render = np.zeros((1,1), dtype=np.int8)
label = np.zeros((1,1), dtype=np.int8)
preview = np.zeros((1,1), dtype=np.int8)
preview_window = 'preview'
output_dir = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/mini patch inspection/'
image_name = ''
#store all regions coordinates that belongs to same image
rois = []

#store prediction paths
sorted_predictions_paths = []

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

def calculate_performance(label, prediction):
    tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
    recall = Statistics.GetRecall(tp, fn)
    precision = Statistics.GetPrecision(tp, fp)
    accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
    f1 = Statistics.GetF1Score(recall, precision)
    IoU = Statistics.GetIoU(label, prediction)
    dice = Statistics.GetDiceCoef(label, prediction)
    accuracy = round(accuracy, 4)
    recall = round(recall, 4)
    precision = round(precision, 4)
    IoU = round(IoU, 4)
    dice = round(dice, 4)
    word = 'acc' + str(accuracy) + '_pre' + str(precision)+ '_rec' + str(recall)+ '_iou' + str(IoU)+ '_dice' + str(dice)
    return word

def save_and_analyze_prediction(path, left, top, right, bottom, image_crop, label_crop, prediction_path):
    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
    prediction_crop = prediction[left:right, top:bottom]
    prediction_crop = cv2.resize(prediction_crop, (50 * 5, 50 * 5),
                            interpolation=cv2.INTER_NEAREST)
    crop_name = '_' + str(left) + '_' + str(top) + '_' + str(right) + '_' + str(bottom)
    outer_color = (0, 200, 200)
    inner_color = (0, 70, 70)
    rendered_crop_prediction_bad = Render.Defects(image_crop, prediction_crop, outer_color, inner_color)
    rendered_crop_prediction_bad = render_border(rendered_crop_prediction_bad)
    name = get_file_name_only(prediction_path)
    cv2.imwrite(path + crop_name + '_' + name + '.bmp', rendered_crop_prediction_bad)

    #draw double mark of label and prediction
    common_color = (0, 130, 0)
    prediction_color = (0, 170, 170)
    label_color = (0, 0, 200)
    common_image = np.copy(image_crop)
    common_image = cv2.cvtColor(common_image, cv2.COLOR_GRAY2RGB)
    width = label_crop.shape[0]
    height = label_crop.shape[1]
    for x in range(0, width):
        for y in range(0, height):
            label_val = label_crop[x,y]
            prediction_val = prediction_crop[x,y]
            if label_val > 0 and prediction_val == 0:
                common_image[x, y, 0] = addPixels(label_color[0], common_image[x, y, 0])
                common_image[x, y, 1] = addPixels(label_color[1], common_image[x, y, 1])
                common_image[x, y, 2] = addPixels(label_color[2], common_image[x, y, 2])
            elif label_val == 0 and prediction_val > 0:
                common_image[x, y, 0] = addPixels(prediction_color[0], common_image[x, y, 0])
                common_image[x, y, 1] = addPixels(prediction_color[1], common_image[x, y, 1])
                common_image[x, y, 2] = addPixels(prediction_color[2], common_image[x, y, 2])
            elif label_val > 0 and prediction_val > 0:
                common_image[x, y, 0] = addPixels(common_color[0], common_image[x, y, 0])
                common_image[x, y, 1] = addPixels(common_color[1], common_image[x, y, 1])
                common_image[x, y, 2] = addPixels(common_color[2], common_image[x, y, 2])
    common_image = render_border(common_image)
    perf_word = calculate_performance(label_crop, prediction_crop)
    cv2.imwrite(path + crop_name + '_' + name + perf_word + '_both.bmp', common_image)

def addPixels(first, second):
    sum = first + second
    if sum > 255:
        sum = 255
    return sum

def save_and_analyze_full_prediction(path, image, label, prediction_path, rois):
    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
    name = get_file_name_only(prediction_path)

    outer_color = (0, 200, 200)
    inner_color = (0, 70, 70)
    rendered_prediction_bad = Render.Defects(image, prediction, outer_color, inner_color)
    rendered_prediction_bad = render_border(rendered_prediction_bad)
    name = get_file_name_only(prediction_path)
    rendered_prediction_bad = render_regions(rendered_prediction_bad, rois)
    cv2.imwrite(path + '_' + name + '.bmp', rendered_prediction_bad)

    #draw double mark of label and prediction
    common_color = (0, 170, 0)
    prediction_color = (0, 130, 130)
    label_color = (0, 0, 200)
    common_image = np.copy(image)
    common_image = cv2.cvtColor(common_image, cv2.COLOR_GRAY2RGB)
    width = label.shape[0]
    height = label.shape[1]
    for x in range(0, width):
        for y in range(0, height):
            label_val = label[x,y]
            prediction_val = prediction[x,y]
            if label_val > 0 and prediction_val == 0:
                common_image[x, y, 0] = addPixels(label_color[0], common_image[x, y, 0])
                common_image[x, y, 1] = addPixels(label_color[1], common_image[x, y, 1])
                common_image[x, y, 2] = addPixels(label_color[2], common_image[x, y, 2])
            elif label_val == 0 and prediction_val > 0:
                common_image[x, y, 0] = addPixels(prediction_color[0], common_image[x, y, 0])
                common_image[x, y, 1] = addPixels(prediction_color[1], common_image[x, y, 1])
                common_image[x, y, 2] = addPixels(prediction_color[2], common_image[x, y, 2])
            elif label_val > 0 and prediction_val > 0:
                common_image[x, y, 0] = addPixels(common_color[0], common_image[x, y, 0])
                common_image[x, y, 1] = addPixels(common_color[1], common_image[x, y, 1])
                common_image[x, y, 2] = addPixels(common_color[2], common_image[x, y, 2])
    common_image = render_border(common_image)
    common_image = render_regions(common_image, rois)
    perf_word = calculate_performance(label, prediction)
    cv2.imwrite(path + '_' + name + perf_word + '_both.bmp', common_image)

def save_crop(path, left, top, right, bottom, image, label, image_crop, label_crop):
    global rois
    global sorted_predictions_paths
    rois.append(left)
    rois.append(top)
    rois.append(right)
    rois.append(bottom)

    outer_color = (0, 0, 200)
    inner_color = (0, 0, 70)

    rendered_image = Render.Defects(image, label, outer_color, inner_color)
    #invert label
    inverted_label = abs(255 - label)

    rendered_image = render_regions(rendered_image, rois)
    rendered_image = render_border(rendered_image)
    save_preview_window = 'image_rois'
    cv2.imshow(save_preview_window, rendered_image)

    #save cropped image and labels, save image 2 times, showing good and bad regions
    #bad region

    rendered_crop_label_bad = Render.Defects(image_crop, label_crop, outer_color, inner_color)
    rendered_crop_label_bad = render_border(rendered_crop_label_bad)
    #good region
    outer_color = (0, 80, 0)
    inner_color = (0, 40, 0)
    # invert cropped label
    inverted_label_crop = abs(255 - label_crop)
    rendered_crop_label_good = Render.Defects(image_crop, inverted_label_crop, outer_color, inner_color)
    rendered_crop_label_good = render_border(rendered_crop_label_good)

    #save region from all predictions
    for prediction_path in sorted_predictions_paths:
        save_and_analyze_prediction(path, left, top, right, bottom, image_crop, label_crop, prediction_path)
        save_and_analyze_full_prediction(path, image, label, prediction_path, rois)

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

def get_file_name_only(path):
    file_name_with_ext = path.rsplit('\\', 1)[1]
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    return file_name


def get_all_defects_images(name_segment, prediction_paths):
    predictions_images_paths = []
    for prediction in prediction_paths:
        prediction_photo_name = get_file_name_only(prediction)
        if name_segment in prediction_photo_name:
            predictions_images_paths.append(prediction)
    return predictions_images_paths

def show_all_predictions(image, predictions_images_paths):
    #show rendered images
    for predictions_images_path in predictions_images_paths:
        prediction_image = cv2.imread(predictions_images_path, cv2.IMREAD_GRAYSCALE)
        name = get_file_name_only(predictions_images_path)
        outer_color = (0, 0, 200)
        inner_color = (0, 0, 80)
        _, prediction_image_th = cv2.threshold(prediction_image, 127, 255, cv2.THRESH_BINARY)
        image_with_render = Render.Defects(image, prediction_image_th, outer_color, inner_color)
        cv2.imshow(name, image_with_render)
        cv2.waitKey(1)

def hide_all_prediction(predictions_images_paths):
    for predictions_images_path in predictions_images_paths:
        name = get_file_name_only(predictions_images_path)
        cv2.destroyWindow(name)

def show_redered_label(image, label, segment_name):
    outer_color = (0, 0, 200)
    inner_color = (0, 0, 80)
    _, label_image_th = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
    image_with_render = Render.Defects(image, label_image_th, outer_color, inner_color)
    cv2.imshow(segment_name + '_label', image_with_render)
    cv2.waitKey(1)

def hide_redered_label(segment_name):
    cv2.destroyWindow(segment_name + '_label')

#################################################
#gather all images

imagesPath = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/mini patch inspection/image/'
labelsPath = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/mini patch inspection/label/'
predictionPath = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/mini patch inspection/prediction/'
image_paths = glob.glob(imagesPath + '*.bmp')
label_paths = glob.glob(labelsPath + '*.bmp')
prediction_paths = glob.glob(predictionPath + '*.bmp')

cv2.namedWindow(preview_window)
cv2.setMouseCallback(preview_window, mouse_event)

#image name segments (ids)
segments = ['16', '27', '28', '34']
for image_path in image_paths:
    #get label image name
    image_name = get_file_name_only(image_path)
    for image_name_segment in segments:
        if image_name_segment in image_name:
            #Open image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #Draw frame
            image = draw_frame(image)
            preview = np.copy(image)
            cv2.imshow('image', image)
            cv2.waitKey(1)
            #Search right label image
            for label_path in label_paths:
                label_name = get_file_name_only(label_path)
                if image_name_segment in label_name:
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    cv2.imshow('label', label)
                    cv2.waitKey(1)

            #Search right prediction image
            for prediction_path in prediction_paths:
                prediction_name = get_file_name_only(prediction_path)
                if image_name_segment in prediction_name:
                    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
                    cv2.waitKey(1)

            sorted_predictions_paths = get_all_defects_images(image_name_segment, prediction_paths)
            show_all_predictions(image, sorted_predictions_paths)

            show_redered_label(image, label, image_name_segment)

            cv2.imshow(preview_window, preview)
            key = cv2.waitKey(0)
            print('Clearing rois...')
            rois.clear()
            hide_all_prediction(sorted_predictions_paths)
            hide_redered_label(image_name_segment)


"""
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

    #render all all possible defect on image


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
"""