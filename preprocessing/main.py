import os
import glob
import cv2
import preprocessing.crop_to_tiles
from preprocessing.augmentation import Augmentation
import preprocessing.image_normalization

def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list

def make_output_directory(output_dir):
    if not os.path.exists(output_dir):
        print('Making output directory: ' + output_dir)
        os.makedirs(output_dir)

def get_image_name(path):
    image_name_with_ext = path.rsplit('\\', 1)[1]
    image_name, image_extension = os.path.splitext(image_name_with_ext)
    return image_name

def check_if_directory_exist(dir):
    if not os.path.exists(dir):
        print(dir + ' - directory does not exist!')
        return False
    else:
        return True

#crop to tile, rotate 90, 180, 270, 0 degrees and flip and rotate again
def augment(image_path, label_path, image_name, output_dir, resize_ratio = 1.0, histogram_norm = False, brightness_correction = False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if histogram_norm:
        image = preprocessing.image_normalization.adaptive_histogram_normalization(image)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if (resize_ratio != 1.0):
        width = int(image.shape[1] * resize_ratio)
        height = int(image.shape[0] * resize_ratio)
        image = cv2.resize(image, (width, height))
        label = cv2.resize(label, (width, height))
    tile_width = 320
    tile_height = 320
    x_overlay = 20
    y_overlay = 20
    image_width = image.shape[1]
    image_height = image.shape[0]
    rois = preprocessing.crop_to_tiles.splitImageToTiles(image_width, image_height, tile_width, tile_height, x_overlay, y_overlay)
    #make output directories for image and label
    images_output = output_dir + 'Images/'
    labels_output = output_dir + 'Labels/'
    make_output_directory(images_output)
    make_output_directory(labels_output)

    for i in range(0, len(rois)):
        roi = rois[i]
        image_roi = preprocessing.crop_to_tiles.cropImageFromRegion(image, roi)
        label_roi = preprocessing.crop_to_tiles.cropImageFromRegion(label, roi)
        rotation_angles = [0.0, 90.0, 180.0, 270.0]
        for rotation_angle in rotation_angles:
            rot_image = Augmentation.RotateImage(image_roi, rotation_angle)
            rot_label = Augmentation.RotateImage(label_roi, rotation_angle)
            angle = (int)(rotation_angle)
            preprocessing.crop_to_tiles.saveImage(rot_image, images_output,
                                                  image_name + '_' + str(i) + '_' + str(angle), '')
            preprocessing.crop_to_tiles.saveImage(rot_label, labels_output,
                                                  image_name + '_' + str(i) + '_' + str(angle), '_label')
            if brightness_correction:
                brightness_corrections = [-15, -5, 15]
                for brightness in brightness_corrections:
                    image_roi_b = Augmentation.BrighnessCorrection(rot_image, brightness)
                    preprocessing.crop_to_tiles.saveImage(image_roi_b, images_output,
                                                          image_name + '_' + str(i) + '_' + str(angle) + str(brightness) + 'br', '')
                    preprocessing.crop_to_tiles.saveImage(rot_label, labels_output,
                                                          image_name + '_' + str(i) + '_' + str(angle) + str(brightness) + 'br', '_label')


        image_roi = Augmentation.FlipImageHorizontally(image_roi)
        label_roi = Augmentation.FlipImageHorizontally(label_roi)
        for rotation_angle in rotation_angles:
            rot_image = Augmentation.RotateImage(image_roi, rotation_angle)
            rot_label = Augmentation.RotateImage(label_roi, rotation_angle)
            angle = (int)(rotation_angle)
            preprocessing.crop_to_tiles.saveImage(rot_image, images_output,
                                                  image_name + '_' + str(i) + '_' + str(angle), '_f')
            preprocessing.crop_to_tiles.saveImage(rot_label, labels_output,
                                                  image_name + '_' + str(i) + '_' + str(angle),'_label_f')
            if brightness_correction:
                brightness_corrections = [-15, -5, 15]
                for brightness in brightness_corrections:
                    image_roi_b = Augmentation.BrighnessCorrection(rot_image, brightness)
                    preprocessing.crop_to_tiles.saveImage(image_roi_b, images_output,
                                                          image_name + '_' + str(i) + '_' + str(angle) + str(brightness) + 'br', '_f')
                    preprocessing.crop_to_tiles.saveImage(rot_label, labels_output,
                                                          image_name + '_' + str(i) + '_' + str(angle) + str(brightness) + 'br', '_label_f')

def save_test_image(image_path, label_path, image_name, output_dir, resize_ratio = 1.0, histogram_norm = False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if histogram_norm:
        image = preprocessing.image_normalization.adaptive_histogram_normalization(image)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if (resize_ratio != 1.0):
        width = int(image.shape[1] * resize_ratio)
        height = int(image.shape[0] * resize_ratio)
        image = cv2.resize(image, (width, height))
        label = cv2.resize(label, (width, height))
    images_output = output_dir + 'Images/'
    labels_output = output_dir + 'Labels/'
    make_output_directory(images_output)
    make_output_directory(labels_output)
    preprocessing.crop_to_tiles.saveImage(image, images_output,
                                          image_name, '')
    preprocessing.crop_to_tiles.saveImage(label, labels_output,
                                          image_name, '_label')

def prepare_crack500(input_dir, output_width, output_height, output_dir):
    if (not check_if_directory_exist(input_dir)):
        return
    #train data
    train_images = gather_image_from_dir(input_dir + 'traindata/')
    mask_prefix = '_mask'
    train_images_list = []
    train_masks_list = []
    for train_image in train_images:
        image_name = get_image_name(train_image)
        #separate masks and images
        if mask_prefix in image_name:
            train_masks_list.append(train_image)
        else:
            train_images_list.append(train_image)
    #test data
    test_images = gather_image_from_dir(input_dir + 'testdata/')
    mask_prefix = '_mask'
    test_images_list = []
    test_masks_list = []
    for test_image in test_images:
        image_name = get_image_name(test_image)
        # separate masks and images
        if mask_prefix in image_name:
            test_masks_list.append(test_image)
        else:
            test_images_list.append(test_image)

    for i in range(0, len(train_images_list)):
        image_path = train_images_list[i]
        label_path = train_masks_list[i]

        image_name = get_image_name(image_path)
        label_name = get_image_name(label_path)

        label_name = label_name.replace('_mask', '')
        if (label_name != image_name):
            raise Exception('Something is wrong in Crack500 dataset!')

        augment(image_path, label_path, image_name + '_crack500', output_dir + 'Train/', 0.25)

    for i in range(0, len(test_images_list)):
        image_path = test_images_list[i]
        label_path = test_masks_list[i]

        image_name = get_image_name(image_path)
        label_name = get_image_name(label_path)

        label_name = label_name.replace('_mask', '')
        if (label_name != image_name):
            raise Exception('Something is wrong in Crack500 dataset!')

        save_test_image(image_path, label_path, image_name + '_crack500', output_dir + 'Test/', 0.25)


def prepare_gaps384(input_dir, output_width, output_height, output_dir, histogram_norm = False):
    if (not check_if_directory_exist(input_dir)):
        return
    train_images_list = []
    train_masks_list = []
    test_images_list = []
    test_masks_list = []

    image_list = gather_image_from_dir(input_dir + 'img/')
    mask_list = gather_image_from_dir(input_dir + 'gt/')

    train_prefix = 'train_'
    test_prefix = 'test_'

    for image in image_list:
        image_name = get_image_name(image)
        if train_prefix in image_name:
            train_images_list.append(image)
        if test_prefix in image_name:
            test_images_list.append(image)

    for mask in mask_list:
        image_name = get_image_name(mask)
        if train_prefix in image_name:
            train_masks_list.append(mask)
        if test_prefix in image_name:
            test_masks_list.append(mask)

    test_images_filtered = []
    #take out image without mask
    for test_mask in test_masks_list:
        test_mask_name = get_image_name(test_mask)
        id_mask = test_mask_name.replace(test_prefix, '')
        id_mask = id_mask.replace('_mask', '')
        for test_image in test_images_list:
            test_image_name = get_image_name(test_image)
            id_image = test_image_name.replace(test_prefix, '')
            if (id_mask == id_image):
                test_images_filtered.append(test_image)
                break

    train_images_filtered = []
    # take out image without mask
    for train_mask in train_masks_list:
        train_mask_name = get_image_name(train_mask)
        id_mask = train_mask_name.replace(train_prefix, '')
        id_mask = id_mask.replace('_mask', '')
        for train_image in train_images_list:
            train_image_name = get_image_name(train_image)
            id_image = train_image_name.replace(train_prefix, '')
            if (id_mask == id_image):
                train_images_filtered.append(train_image)
                break

    #gather filtered out image
    #train
    train_images_filtered_out = []
    for train_image in train_images_list:
        is_filtered_out = True
        for train_image_filtered in train_images_filtered:
            if (train_image == train_image_filtered):
                is_filtered_out = False
                break
        if (is_filtered_out):
            train_images_filtered_out.append(train_image)
    #test
    test_images_filtered_out = []
    for test_image in test_images_list:
        is_filtered_out = True
        for test_image_filtered in test_images_filtered:
            if (test_image == test_image_filtered):
                is_filtered_out = False
                break
        if (is_filtered_out):
            test_images_filtered_out.append(test_image)
    ratio = 0.5
    #augment and crop
    for i in range(0, len(train_images_filtered)):
        #resize 2 times
        image_path = train_images_filtered[i]
        label_path = train_masks_list[i]
        image_name = get_image_name(image_path)
        #take out prefix
        image_name = image_name.replace('train_', '')
        label_name = get_image_name(label_path)
        label_name = label_name.replace('train_', '')
        label_name = label_name.replace('_mask', '')
        if (label_name != image_name):
            raise Exception('Something is wrong in Gaps384 dataset!')
        augment(image_path, label_path, image_name + '_gaps384', output_dir + 'Train/', ratio, histogram_norm)

    for i in range(0, len(test_images_filtered)):
        #resize 2 times
        image_path = test_images_filtered[i]
        label_path = test_masks_list[i]
        image_name = get_image_name(image_path)
        #take out prefix
        image_name = image_name.replace('test_', '')
        label_name = get_image_name(label_path)
        label_name = label_name.replace('test_', '')
        label_name = label_name.replace('_mask', '')
        if (label_name != image_name):
            raise Exception('Something is wrong in Gaps384 dataset!')

        save_test_image(image_path, label_path, image_name + '_gaps384', output_dir + 'Test/', ratio, histogram_norm)

    h = 4

def prepare_crackForest(input_dir, output_width, output_height, output_dir, brightness_correction = False):
    if (not check_if_directory_exist(input_dir)):
        return
    input_dir += 'datasets/Set_0/'
    train_images_list = gather_image_from_dir(input_dir + 'Train/Images/')
    train_labels_list = gather_image_from_dir(input_dir + 'Train/Labels/')
    test_images_list = gather_image_from_dir(input_dir + 'Test/Images/')
    test_labels_list = gather_image_from_dir(input_dir + 'Test/Labels/')
    for i in range(0, len(train_images_list)):
        image_path = train_images_list[i]
        label_path = train_labels_list[i]
        image_name = get_image_name(image_path)
        label_name = get_image_name(label_path)
        label_name = label_name.replace('_label', '')
        if (label_name != image_name):
            raise Exception('Something is wrong in CrackForest dataset!')
        augment(image_path, label_path, image_name + '_crackforest', output_dir + 'Train/', 1, brightness_correction=brightness_correction)

    for i in range(0, len(test_images_list)):
        image_path = test_images_list[i]
        label_path = test_labels_list[i]
        image_name = get_image_name(image_path)
        label_name = get_image_name(label_path)
        label_name = label_name.replace('_label', '')
        if (label_name != image_name):
            raise Exception('Something is wrong in CrackForest dataset!')

        save_test_image(image_path, label_path, image_name + '_crackforest', output_dir + 'Test/', 1)

    h = 4

def main():
    width = 320
    height = 320
    #prepare_crackForest('D:/pavement defect data/CrackForestdatasets/', width, height, 'C:/Users/Rytis/Desktop/datasets/CrackForestdatasets_brightness_output/', brightness_correction=True)
    prepare_gaps384('D:/pavement defect data/GAPs384_raw_img_gt/', width, height,'C:/Users/Rytis/Desktop/datasets/GAPs384_output_0.5percent_size_norm_clipLim01/', histogram_norm=True)
    #prepare_crack500('D:/pavement defect data/crack500/', width, height, 'C:/Users/Rytis/Desktop/crack500_out_0.25percent_size/')

if __name__ == '__main__':
    main()