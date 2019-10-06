import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.utils.vis_utils import plot_model

def dice_loss(y_true, y_pred):
	smooth = 1e-6
	y_true_f = keras.flatten(y_true)
	y_pred_f = keras.flatten(y_pred)
	intersection = keras.sum(y_true_f * y_pred_f)
	answer = (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
	return answer
	
def IOU_calc_loss(y_true, y_pred):
	return 1 - dice_loss(y_true, y_pred)

def unet_3layer(pretrained_weights = None,input_size = (512,512,1)):
	features = 8
	inputs = Input(input_size)
	conv1 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	
	conv2 = Conv2D(features * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(features * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	
	conv3 = Conv2D(features * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(features * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	conv3 = BatchNormalization()(conv3)

	up4 = Conv2D(features * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
	merge4 = concatenate([conv2,up4], axis = 3)
	conv4 = Conv2D(features * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(features * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	conv4 = BatchNormalization()(conv4)

	up5 = Conv2D(features, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
	merge5 = concatenate([conv1,up5], axis = 3)
	conv5 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	conv5 = BatchNormalization()(conv5)
	conv6 = Conv2D(1, 1, activation = 'sigmoid')(conv5)

	model = Model(input = inputs, output = conv6)

	model.compile(optimizer = Adam(lr = 1e-4), loss = IOU_calc_loss, metrics = [dice_loss])
	
	model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

def unet_2layerWithoutBatchNormStride2(pretrained_weights = None,input_size = (480,320,1)):
	features = 32
	inputs = Input(input_size)
	conv1 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	conv1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	
	conv2 = Conv2D(features * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(features * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	conv2 = Dropout(0.5)(conv2)

	up3 = Conv2D(features, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
	merge3 = concatenate([conv1,up3], axis = 3)
	conv3 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
	conv3 = Conv2D(features, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	conv3 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	conv4 = Conv2D(1, 1, activation = 'sigmoid')(conv3)

	model = Model(input = inputs, output = conv4)
	model.compile(optimizer = Adam(lr = 1e-4), loss = IOU_calc_loss, metrics = [dice_loss]) 
	model.summary()
	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

	return model

#UNet with some additional parameters, connection and so on....
def autoEncoder(pretrained_weights = None,
				input_size = (512,512),
				number_of_layers = 2,
				kernel_size = 3,
				number_of_kernels = 8,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				dropout = False,
				dropout_rate = 0.5,
				batch_norm_bottleNeck = True,
				dropout_bottleNeck = True,
				residual_connections = False):

	# Input
	inputs = Input(input_size)
	# Number of feature kernels is equal to feature count passes to function 
	first_layer_features = number_of_kernels
	# Double convolution according to U-Net structure
	conv1 = Conv2D(first_layer_features, kernel_size = (kernel_size, kernel_size), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	# Batch-normalization on demand
	if batch_norm == True:
		conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(first_layer_features, kernel_size = (kernel_size, kernel_size), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	# Dropout on demand
	if dropout == True:
		conv1 = Dropout(dropout_rate)(conv1)
	# Max-pool on demand
	if max_pool == True:
		first_layer = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv1)
	else:
		first_layer = conv1

	


	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)

	#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

	return model