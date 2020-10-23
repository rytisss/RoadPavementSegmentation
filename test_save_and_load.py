import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models.autoencoder import *
from models.losses import Loss
from models.utilities import trainGenerator
import os


###############################
# Just inspect if model trains, saves and loads
###############################

def train():
    number_of_samples = 10
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 1
    tf.keras.backend.clear_session()
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = number_of_samples / batch_size
    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 1
    # Define model
    model = UNet4_First5x5_FirstDeformable(number_of_kernels=8,
                                    input_size = (320,320,1),
                                    loss_function = Loss.CROSSENTROPY50DICE50,
                                    learning_rate=1e-3,
                                    useLeakyReLU=True)

    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'C:\Users\AR\Desktop\freda holes data 2020-10-14/'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    generator = trainGenerator(batch_size, data_dir, 'Image_rois', 'Label_rois', data_gen_args, save_to_dir = None, target_size = (320,320))

    weights_output_dir = r'C:\Users\AR\Desktop\freda holes data 2020-10-14\test_weights/'

    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    weights_output_name = 'UNet4_First5x5_FirstDeformable'

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + ".hdf5"
    # Make checkpoint for saving each
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    model.fit(generator,steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint], shuffle = True)

    # load model
    trained_model = UNet4_First5x5_FirstDeformable(number_of_kernels=2,
                           input_size=(320, 320, 1),
                           loss_function=Loss.CROSSENTROPY50DICE50,
                           learning_rate=1e-3,
                           useLeakyReLU=True,
                            pretrained_weights=weights_output_dir + weights_output_name + '.hdf5')

    print('Saved and loaded successfully!')

def main():

    train()

if __name__ == "__main__":
    main()



