from keras.callbacks import ModelCheckpoint
from models.autoencoder import UNet4, UNet4_res, UNet4_res_asppWF, UNet4_res_aspp, UNet4_res_asppWF_AG, UNet4_res_aspp_AG, UNet5_res, UNet5_res_aspp
from models.losses import Loss
from models.utilities import trainGenerator
import os
###############################
# Super-basic training routine
###############################

def train():
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = 328
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 4
    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 50

    # Define model
    model = UNet4(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    model = UNet5_res_aspp(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50,
                  learning_rate=1e-3)
    model = UNet5_res(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50,
                           learning_rate=1e-3)
    #model = UNet4_res(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_aspp(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_asppWF(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_asppWF_AG(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_aspp_AG(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'C:\Users\DeepLearningRig\Desktop\datasets\CrackForestdatasets_output\Train/'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    generator = trainGenerator(batch_size, data_dir, 'Images', 'Labels', data_gen_args, save_to_dir = None, target_size = (320,320))

    # Directory for weight saving (creates if it does not exist)
    weights_output_dir = 'C:/Users/DeepLearningRig/Desktop/UNet4/'
    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + "My_weights_UNet4-{epoch:03d}-{loss:.4f}.hdf5"

    # Make checkpoint for saving each
    model_checkpoint = ModelCheckpoint(weights_name, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    model.fit_generator(generator,steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint], shuffle = True)

def main():
    train()

if __name__ == "__main__":
    main()



