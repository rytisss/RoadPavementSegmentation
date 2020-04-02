from keras import Model
from keras.callbacks import ModelCheckpoint
from autoencoder import UNet5, UNet5_res, UNet5_res_aspp, UNet5_aspp, UNet4, UNet4_res, UNet4_res_aspp, UNet4_aspp, UNet4_res_asppWF, UNet4_res_aspp, UNet4_res_asppWF_AG
from utilities import *
import os
from losses import *

from models.losses import Loss
from models.utilities import trainGenerator


def train():
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

    generator = trainGenerator(1, data_dir + 'Train/','Images','Labels',None,save_to_dir = None, target_size = (320,320))
    outputDir = 'C:/Users/Rytis/Desktop/CrackForest_UNet4_res/'

    if not os.path.exists(outputDir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputDir)

    outputPath = outputDir + "CrackForest_UNet5-{epoch:03d}-{loss:.4f}.hdf5"
    model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    model.fit_generator(generator,steps_per_epoch=164,epochs=50,callbacks=[model_checkpoint], shuffle = True)

def main():
    train()

if __name__ == "__main__":
    main()



