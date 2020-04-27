from keras.callbacks import ModelCheckpoint
from models.autoencoder import UNet4, UNet4_res, UNet4_res_asppWF, UNet4_res_aspp, UNet4_res_asppWF_AG, UNet4_res_aspp_AG, UNet5_res_aspp, UNet5_res_aspp_First5x5, UNet5_res_First5x5, UNet5_First5x5, UNet4_asspWF, UNet4_assp, UNet4_assp_First5x5, UNet4_asspWF_First5x5, UNet4_assp_AG_First5x5, UNet4_asspWF_AG_First5x5
from models.losses import Loss
from models.utilities import trainGenerator
import os
import keras
###############################
# Super-basic training routine
###############################

# Directory for weight saving (creates if it does not exist)
weights_output_dir = r'D:/drill investigation/data/UNet5_First5x5/'
weights_output_name = 'DrillSegmentation_UNet5_First5x5'

class CustomSaver(keras.callbacks.Callback):
    def __init__(self):
        self.overallIteration = 0
    def on_batch_end(self, iteration, logs={}):
        self.overallIteration += 1
        if self.overallIteration % 1000 == 0 and self.overallIteration != 0:  # or save after some epoch, each k-th epoch etc.
            print('Saving iteration ' + str(self.overallIteration))
            self.model.save(weights_output_dir + weights_output_name + "_{}.hdf5".format(self.overallIteration))

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // 10
    init_lr = 0.001
    lr = init_lr / 2**step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr

def train():
    # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
    number_of_iteration = 5843
    # batch size. How many samples you want to feed in one iteration?
    batch_size = 4
    # number_of_epoch. How many epoch you want to train?
    number_of_epoch = 15

    # Define model
    model = UNet5_First5x5(number_of_kernels=16,input_size = (480,480,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    model = UNet4_assp_AG_First5x5(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    model = UNet4_asspWF_AG_First5x5(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_asppWF(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_asppWF_AG(number_of_kernels=32, input_size=(320, 320, 1), loss_function=Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    #model = UNet4_res_aspp_AG(number_of_kernels=32,input_size = (320,320,1), loss_function = Loss.CROSSENTROPY50DICE50, learning_rate=1e-3)
    # Where is your data?
    # This path should point to directory with folders 'Images' and 'Labels'
    # In each of mentioned folders should be image and annotations respectively
    data_dir = r'D:\drill investigation\data\DrillHolesAugm/'

    # Possible 'on-the-flight' augmentation parameters
    data_gen_args = dict(rotation_range=0.0,
                         width_shift_range=0.00,
                         height_shift_range=0.00,
                         shear_range=0.00,
                         zoom_range=0.00,
                         horizontal_flip=False,
                         fill_mode='nearest')

    # Define data generator that will take images from directory
    generator = trainGenerator(batch_size, data_dir, 'images', 'labels', data_gen_args, save_to_dir = None, target_size = (480,480))

    if not os.path.exists(weights_output_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(weights_output_dir)

    # Define template of each epoch weight name. They will be save in separate files
    weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
    # Custom saving
    saver = CustomSaver()
    # Learning rate scheduler
    learning_rate_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    # Make checkpoint for saving each
    model_checkpoint = ModelCheckpoint(weights_name, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)
    model.fit_generator(generator,steps_per_epoch=number_of_iteration,epochs=number_of_epoch,callbacks=[model_checkpoint, saver, learning_rate_scheduler], shuffle = True)

def main():
    train()

if __name__ == "__main__":
    main()



