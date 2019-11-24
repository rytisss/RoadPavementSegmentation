from keras import Model
from keras.callbacks import ModelCheckpoint
from script.model.autoencoder import *
from script.model.utilities import *
import numpy as np 
import os
import keras
import math

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
from tfdeterminism import patch
patch()

import random
os.environ['PYTHONHASHSEED']=str(1)
random.seed(1)
np.random.seed(1)

data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.00,
                    horizontal_flip=False,
                    fill_mode='nearest')

learningRate = 0.001
kernels = 32
setNumber = 0

#outputDir = 'E:/RoadCracksInspection/trainingOutput/Set_' + str(setNumber) + '/l4k' + str(kernels) + 'AutoEncoder4_5x5WeightCross' + str(learningRate) + '_' + str(setNumber) +'/'
outputDir = 'C:/src/Set_' + str(setNumber) + '/l4k' + str(kernels) + 'AutoEncoder4_5x5WeightCross' + str(learningRate) + '_' + str(setNumber)+'/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)
generator = trainGenerator(4,'E:/RoadCracksInspection/datasets/Set_' + str(setNumber) + '/Train/AUGM/','Images','Labels',data_gen_args,save_to_dir = None, target_size = (320,480))
model = AutoEncoder4_5x5(number_of_kernels=kernels,input_size = (320,480,1), loss_function = Loss.DICE)
outputPath = outputDir + "AutoEncoder4_5x5-{epoch:03d}-{loss:.4f}.hdf5"
#scheduler = AlphaScheduler()
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False)

model.fit_generator(generator,steps_per_epoch=82,epochs=50,callbacks=[model_checkpoint])
keras.backend.clear_session()

