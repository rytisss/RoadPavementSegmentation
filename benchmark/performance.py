from models.autoencoder import UNet4_res_asppWF_AG, UNet4_res_asppWF, UNet4_res_aspp_AG, UNet4_res_aspp, UNet4_aspp, \
    UNet4_res, UNet4
from utilities import *
import glob
from autoencoder import *
import preprocessing.crop_to_tiles
import cv2
import keras
import time

def testCalculationPerformance(model, image):
    model.summary()
    image_norm = image / 255
    image_norm = np.reshape(image_norm, image_norm.shape + (1,))
    image_norm = np.reshape(image_norm, (1,) + image_norm.shape)
    sum = 0.0
    counter = 0
    for i in range(0, 2):
        start = time.time()
        model.predict(image_norm)
        end = time.time()
        #skip first iteration
        duration = end - start
        #print('Prediction duration: ' + str(duration))
        if counter > 0:
            sum += duration
        counter += 1
    averageDuration = sum / (counter - 1)
    print('Average prediction duration: ' + str(averageDuration))
    keras.backend.clear_session()

def main():
    imagePath = '0417_gaps384_6_180.jpg'
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    print('UNet4_res_asppWF_AG')
    model = UNet4_res_asppWF_AG(number_of_kernels=32, input_size=(320, 320, 1),
                                loss_function=Loss.CROSSENTROPY50DICE50)
    testCalculationPerformance(model, image)

    print('UNet4_res_asppWF')
    model = UNet4_res_asppWF(number_of_kernels=32, input_size=(320, 320, 1),
                             loss_function=Loss.CROSSENTROPY50DICE50)
    testCalculationPerformance(model, image)

    print('UNet4_res_aspp_AG')
    model = UNet4_res_aspp_AG(number_of_kernels=32, input_size=(320, 320, 1),
                             loss_function=Loss.CROSSENTROPY50DICE50)
    testCalculationPerformance(model, image)

    print('UNet4_res_aspp')
    model = UNet4_res_aspp(number_of_kernels=32, input_size=(320, 320, 1),
                             loss_function=Loss.CROSSENTROPY50DICE50)
    testCalculationPerformance(model, image)

    print('UNet4_res')
    model = UNet4_res(number_of_kernels=32, input_size=(320, 320, 1),
                             loss_function=Loss.CROSSENTROPY50DICE50)
    testCalculationPerformance(model, image)

    print('UNet4')
    model = UNet4(number_of_kernels=32, input_size=(320, 320, 1),
                             loss_function=Loss.CROSSENTROPY50DICE50)
    testCalculationPerformance(model, image)

if __name__ == '__main__':
    main()