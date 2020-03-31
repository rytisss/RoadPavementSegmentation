import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint
from autoencoder import *
from utilities import *
import numpy as np 
import os
import keras
from losses import *
import math

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
setNumber = 1

#outputDir = 'E:/RoadCracksInspection/trainingOutput/Set_' + str(setNumber) + '/l4k' + str(kernels) + 'AutoEncoder4_5x5WeightCross' + str(learningRate) + '_' + str(setNumber) +'/'
outputDir = 'C:/Users/DeepLearningRig/Desktop/weights_Set0/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(1, 'C:/Users/DeepLearningRig/Desktop/datasets/Set_0/Train/AUGM/','Images','Labels',data_gen_args,save_to_dir = None, target_size = (320,480))
model = AutoEncoder4ResAddOpConcDec(input_size = (320,480,1))
outputPath = outputDir + "AutoEncoder4_5x5-{epoch:03d}-{loss:.4f}.hdf5"
#scheduler = AlphaScheduler()
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False, save_weights_only=False, shuffle = True)

model.fit_generator(generator,steps_per_epoch=82,epochs=50,callbacks=[model_checkpoint])
keras.backend.clear_session()

