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
    model_def = UNet4_First5x5_BothDeformable(number_of_kernels=8,
                                    input_size = (320,320,1),
                                    loss_function = Loss.CROSSENTROPY50DICE50,
                                    learning_rate=1e-3,
                                    useLeakyReLU=True)

    model = UNet4_First5x5(number_of_kernels=8,
                                           input_size=(320, 320, 1),
                                           loss_function=Loss.CROSSENTROPY50DICE50,
                                           learning_rate=1e-3,
                                           useLeakyReLU=True)

    for layer in zip(model.layers, model_def.layers):
        first_weights = layer[0].get_weights()
        second_weights = layer[1].get_weights()
        # check types: first should be conv2d, second - deformable conv2d
        name_1 = layer[0].__class__.__name__
        name_2 = layer[1].__class__.__name__
        if name_1 == 'Conv2D' and name_2 == 'DeformableConv2D':
            # extract kernels from first weights and put it to second
            first_shape = first_weights[0].shape
            # iterate thought second and third axis of first shape
            for ind_2 in range(0, first_shape[2]):
                for ind_3 in range(0, first_shape[3]):
                    first_weight_kernel = first_weights[0][:,:,ind_2,ind_3]
                    # place in deformable convolution2d weights
                    index_in_deformable_conv2d = ind_2 * first_shape[3] + ind_3
                    print(index_in_deformable_conv2d)
                    # adjust weights in deformed convolution
                    second_weights[0][:,:,index_in_deformable_conv2d,0] = first_weight_kernel
                    print('\n First:')
                    print(first_weight_kernel)
                    print('\n Second:')
                    print(second_weights[0][:,:,index_in_deformable_conv2d,0])
                    layer[1].set_weights(second_weights)
        if name_1 == 'BatchNormalization' and name_2 == 'BatchNormalization':
            layer[1].set_weights(layer[0].get_weights())
        if name_1 == 'Conv2D' and name_2 == 'Conv2D':
            layer[1].set_weights(layer[0].get_weights())


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



