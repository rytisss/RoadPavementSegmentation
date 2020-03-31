from keras import Model
from keras.callbacks import ModelCheckpoint
from autoencoder import UNet5, UNet5_res, UNet5_res_aspp, UNet5_aspp, UNet4, UNet4_res, UNet4_res_aspp, UNet4_aspp, UNet4_res_asppWF, UNet4_res_aspp, UNet4_res_asppWF_AG
from utilities import *
import numpy as np 
import os
from losses import *
import keras
import math
import random
from models.predict_by_patches import predict_by_patches
from benchmark.analyzeAll import AnalyzeArchitecture


data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.00,
                    horizontal_flip=False,
                    fill_mode='nearest')


outputDir = 'C:/Users/Rytis/Desktop/CrackForest_UNet4_res/'

class CustomSaver(keras.callbacks.Callback):
    def __init__(self):
        self.iteration = 0
    def on_batch_end(self, iteration, logs={}):
        self.iteration += 1
        print('\n' + str(iteration) + ' ' + str(self.iteration) + '\n')
        #if epoch == 2:  # or save after some epoch, each k-th epoch etc.
        #    self.model.save("model_{}.hd5".format(epoch))

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // 10
    init_lr = 0.001
    lr = init_lr / 2**step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr

def train_and_test():
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    #model = UNet4_2bottleneck(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50)
    #model = UNet4_res(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50)
    #model = UNet4_res_aspp(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50)
    #model = UNet4_res_asppWF(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50)
    #model = UNet4_res_asppWF_AG(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50)
    #model = UNet4_res_dense_aspp(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50)
    #model = UNet4_res_aspp(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50)

    #outputDir = 'E:/RoadCracksInspection/trainingOutput/Set_' + str(setNumber) + '/l4k' + str(kernels) + 'AutoEncoder4_5x5WeightCross' + str(learningRate) + '_' + str(setNumber) +'/'
    model = UNet4_aspp(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50)
    data_dir = 'C:/Users/Rytis/Desktop/CrackForestdatasets_output/'
    if not os.path.exists(outputDir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputDir)
    generator = trainGenerator(1, data_dir + 'Train/','Images','Labels',data_gen_args,save_to_dir = None, target_size = (320,320))
    outputPath = outputDir + "CrackForest_UNet5-{epoch:03d}-{loss:.4f}.hdf5"
    model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    #saver = CustomSaver()
    model.fit_generator(generator,steps_per_epoch=164,epochs=50,callbacks=[model_checkpoint, lr_callback, saver], shuffle = True)
    keras.backend.clear_session()
    #prediction
    predict_by_patches(outputDir, data_dir + 'Test/', True)
    cv2.destroyAllWindows()
    #statistics
    AnalyzeArchitecture(outputDir + 'output/', data_dir + 'Test/')
    cv2.destroyAllWindows()

def main():
    train_and_test()

if __name__ == "__main__":
    main()



