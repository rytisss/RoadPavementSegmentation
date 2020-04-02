from models.utilities import *
import glob
from models.autoencoder import *
import preprocessing.crop_to_tiles
import cv2
import keras
import os

def gather_image_from_dir(input_dir):
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list

def get_file_name(path):
    file_name_with_ext = path.rsplit('\\', 1)[1]
    file_name, file_extension = os.path.splitext(file_name_with_ext)
    return file_name

def predict_by_patches():
    # Tile/patch/region size
    input_size = (320, 320)

    # Weights path
    weight_path = r'C:\Users\Rytis\Desktop\pavement_defect_results\pretrained_UNet4_res_aspp_AG\gaps384\Gaps384_pretrained_UNet4_res_aspp_AG_750.hdf5'

    # Choose your 'super-model'
    model = UNet4_res_aspp_AG(pretrained_weights=weight_path, number_of_kernels=32, input_size=(320, 320, 1),
                              loss_function=Loss.CROSSENTROPY50DICE50)

    # Test images directory
    test_images = r'C:\Users\Rytis\Desktop\CrackForestdatasets_output\Test\Images/'

    image_paths = gather_image_from_dir(test_images)

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        width = image.shape[1]
        height = image.shape[0]

        # Tile/patch/region overlay
        overlay = 20

        # Get image regions - tiles
        rois = preprocessing.crop_to_tiles.splitImageToTiles(width, height, input_size[0], input_size[1], overlay,
                                                                 overlay)
        # Create image for output
        prediction = np.zeros_like(image)

        for roi in rois:
            # Get tile/patch/region
            image_crop = preprocessing.crop_to_tiles.cropImageFromRegion(image, roi)
            # Preprocess
            image_crop_norm = image_crop / 255
            image_crop_norm = np.reshape(image_crop_norm, image_crop_norm.shape + (1,))
            image_crop_norm = np.reshape(image_crop_norm, (1,) + image_crop_norm.shape)
            # Predict
            prediction_crop_norm = model.predict(image_crop_norm)
            # Normalize to image
            prediction_crop = prediction_crop_norm[0, :, :, 0]
            prediction_crop *= 255
            prediction_crop = prediction_crop.astype(np.uint8)
            # Put back to original image with OR operation
            prediction[roi[1]:roi[3], roi[0]:roi[2]] = cv2.bitwise_or(prediction[roi[1]:roi[3], roi[0]:roi[2]],
                                                                        prediction_crop)
            # Do you want to visualize image?
            show_image = True
            if show_image:
                cv2.imshow("image", image_crop)
                cv2.imshow("prediction", prediction_crop)
                cv2.imshow("full prediction", prediction)
                cv2.imshow("full image", image)
                cv2.waitKey(1)



def main():
    predict_by_patches()

if __name__ == '__main__':
    main()