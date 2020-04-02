from models.utilities import *
import glob
from models.autoencoder import *
import cv2

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

##########################################
# Super-basic testing/prediction routine
##########################################

def predict():
    # Weights path
    weight_path = r'C:\Users\Rytis\Desktop\pavement_defect_results\pretrained_UNet4_res_aspp_AG\gaps384\Gaps384_pretrained_UNet4_res_aspp_AG_750.hdf5'

    # Choose your 'super-model'
    model = UNet4_res_aspp_AG(pretrained_weights=weight_path, number_of_kernels=32, input_size=(320, 320, 1),
                              loss_function=Loss.CROSSENTROPY50DICE50)

    # Test images directory
    test_images = r'C:\Users\Rytis\Desktop\CrackForestdatasets_output\Train\Images/'

    image_paths = gather_image_from_dir(test_images)

    # Load and predict on all images from directory
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # preprocess
        image_norm = image / 255
        image_norm = np.reshape(image_norm, image_norm.shape + (1,))
        image_norm = np.reshape(image_norm, (1,) + image_norm.shape)
        # predict
        prediction = model.predict(image_norm)
        # normalize to image
        prediction_image_norm = prediction[0, :, :, 0]
        prediction_image = prediction_image_norm * 255
        prediction_image = prediction_image.astype(np.uint8)

        # Do you want to visualize image?
        show_image = True
        if show_image:
            cv2.imshow("image", image)
            cv2.imshow("prediction", prediction_image)
            cv2.waitKey(1)

def main():
    predict()

if __name__ == '__main__':
    main()