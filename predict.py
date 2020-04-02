from models.autoencoder import AutoEncoder4VGG16_ASPP, AutoEncoder4_5x5ASPP, AutoEncoder4VGG16_5x5_ASPP
from utilities import *
import glob
from autoencoder import *
import preprocessing.crop_to_tiles
import cv2
import keras

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

def predict_by_patches(weight_path = '', test_data_dir = '', show_output = False):
    input_size = (320, 320)
    #weight_path = "E:/pavement inspection/lr_scheduler/CrackForest_UNet5_res_aspp/"
    weights = glob.glob(weight_path + '*.hdf5')

    prediction_image_output_dir = weight_path + 'output/'

    for weight in weights:
        print(weight)
        if 'UNet4_res_asppWF_AG' in weight:
            model = UNet4_res_asppWF_AG(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                        loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_res_asppWF' in weight:
            model = UNet4_res_asppWF(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                     loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_res_aspp_AG' in weight:
            model = UNet4_res_aspp_AG(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                      loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_res_aspp' in weight:
            model = UNet4_res_aspp(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                   loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_res_dense_aspp' in weight:
            model = UNet4_res_dense_aspp(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                         loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_res_aspp' in weight:
            model = UNet4_res_aspp(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                   loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_aspp' in weight:
            model = UNet4_aspp(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                               loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet5_res_aspp' in weight:
            model = UNet5_res_aspp(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                                   loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet5_aspp' in weight:
            model = UNet5_aspp(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                               loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4_res' in weight:
            model = UNet4_res(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                              loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet4' in weight:
            model = UNet4(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                          loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet5_res' in weight:
            model = UNet5_res(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                              loss_function=Loss.CROSSENTROPY50DICE50)
        elif 'UNet5' in weight:
            model = UNet5(pretrained_weights=weight, number_of_kernels=32, input_size=(320, 320, 1),
                          loss_function=Loss.CROSSENTROPY50DICE50)
        else:
            raise Exception('Unindentified model!')

        test_images = test_data_dir + 'Images/' #test data directory, we are interest only in images

        image_paths = gather_image_from_dir(test_images)
        overlay = 20 #10 overlay in tiles

        #saving directory form
        weight_name = get_file_name(weight)
        weight_output_dir = prediction_image_output_dir + weight_name + '/'
        if not os.path.exists(weight_output_dir):
            print('Output directory doesnt exist!\n')
            print('It will be created in ' + weight_output_dir + '\n')
            os.makedirs(weight_output_dir)

        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            width = image.shape[1]
            height = image.shape[0]
            rois = preprocessing.crop_to_tiles.splitImageToTiles(width, height, input_size[0], input_size[1], overlay,
                                                                 overlay)
            #create image for output
            prediction = np.zeros_like(image)
            image_name = get_file_name(image_path)
            for roi in rois:
                image_crop = preprocessing.crop_to_tiles.cropImageFromRegion(image, roi)
                #preprocess
                image_crop_norm = image_crop / 255
                image_crop_norm = np.reshape(image_crop_norm, image_crop_norm.shape + (1,))
                image_crop_norm = np.reshape(image_crop_norm, (1,) + image_crop_norm.shape)
                #predict
                prediction_crop_norm = model.predict(image_crop_norm)
                #normalize to image
                prediction_crop = prediction_crop_norm[0, :, :, 0]
                prediction_crop *= 255
                prediction_crop = prediction_crop.astype(np.uint8)
                # put back to original image with OR operation
                prediction[roi[1]:roi[3], roi[0]:roi[2]] = cv2.bitwise_or(prediction[roi[1]:roi[3], roi[0]:roi[2]],
                                                                          prediction_crop)
                if show_output:
                    cv2.imshow("image", image_crop)
                    cv2.imshow("prediction", prediction_crop)
                    cv2.imshow("full prediction", prediction)
                    cv2.imshow("full image", image)
                    cv2.waitKey(1)

            cv2.imwrite(weight_output_dir + image_name + '.jpg', prediction)

        keras.backend.clear_session()

def main():
    predict_by_patches()

if __name__ == '__main__':
    main()