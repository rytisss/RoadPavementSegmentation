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

def EncodingLayer(input,
				kernel_size = 3,
				kernels = 8,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				residual_connections = False,
				isInput = False):

	if residual_connections == True:
		# calculate how many times
		downscale = stride
		if max_pool == True:
			downscale *= max_pool_size
		shortcut = Conv2D(kernels, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(input)
		if max_pool == True:
			shortcut = MaxPooling2D(pool_size=(downscale, downscale))(shortcut) 
		
		#if batch_norm == True and isInput == False:
		#	shortcut = BatchNormalization()(shortcut)
	# do not make batch-normalization on the first layer/neural network input, because data here is already normalized!
	if batch_norm == True and isInput == False:
		input = BatchNormalization()(input)
	# do not activate in first layer
	if isInput == False:
		input = Activation('relu')(input)
	# Double convolution according to U-Net structure
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = stride, padding = 'same', kernel_initializer = 'he_normal')(input)
	# Batch-normalization on demand
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	# Max-pool on demand
	if max_pool == True:
		conv = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
	# in case we are using residual connection, add shortcut
	if residual_connections == True:
		conv = Add()([conv, shortcut])
	#in next step this output needs to be activated
	output = conv
	return output

def DecodingLayer(input,
				skippedInput,
				upSampleSize = 2,
				kernel_size = 3,
				kernels = 8,
				batch_norm = True,
				residual_connections = False):

	concatenatedInput = Concatenate()([input, skippedInput])
	upsampledInput = UpSampling2D((upSampleSize, upSampleSize))(concatenatedInput)
	
	if residual_connections == True:
		shortcut = Conv2D(kernels, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(upsampledInput)
		#if batch_norm == True:
		#	shortcut = BatchNormalization()(shortcut)
	if batch_norm == True:
		upsampledInput = BatchNormalization()(upsampledInput)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(upsampledInput)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(upsampledInput)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	if residual_connections == True:
		conv = Add()([conv, shortcut])
	output = conv
	return output

#5-layer UNet with residual connection
def AutoEncoderRes5(pretrained_weights = None,
				input_size = (480,320,1),
				number_of_layers = 2,
				kernel_size = 3,
				number_of_kernels = 16,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				batch_norm_bottleNeck = True,
				residual_connections = False):
	# Input
	inputs = Input(input_size)
	#encoding
	enc0 = EncodingLayer(inputs, kernel_size, number_of_kernels, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections, isInput = True)
	enc1 = EncodingLayer(enc0, kernel_size, number_of_kernels * 2, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections)
	enc2 = EncodingLayer(enc1, kernel_size, number_of_kernels * 4, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections)
	enc3 = EncodingLayer(enc2, kernel_size, number_of_kernels * 8, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections)
	#bottleneck without residual (might be without batch-norm)
	enc4 = EncodingLayer(enc3, kernel_size, number_of_kernels * 16, stride, False, max_pool_size, batch_norm, residual_connections = False)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec3 = DecodingLayer(enc4, enc3, 2, kernel_size, number_of_kernels * 8, batch_norm, residual_connections = residual_connections)
	dec2 = DecodingLayer(dec3, enc2, 2, kernel_size, number_of_kernels * 4, batch_norm, residual_connections = residual_connections)
	dec1 = DecodingLayer(dec2, enc1, 2, kernel_size, number_of_kernels * 2, batch_norm, residual_connections = residual_connections)
	dec0 = DecodingLayer(dec1, enc0, 2, kernel_size, number_of_kernels, batch_norm, residual_connections = residual_connections)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(dec0)
	model = Model(inputs, outputs)
	model.compile(optimizer = Adam(lr = 1e-4), loss = IOU_calc_loss, metrics = [dice_loss])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes5.png', show_shapes=True, show_layer_names=True)
	return model

#5-layer UNet without residual connection
def AutoEncoder5(pretrained_weights = None,
				input_size = (480,320,1),
				number_of_layers = 2,
				kernel_size = 3,
				number_of_kernels = 8,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				batch_norm_bottleNeck = True,
				residual_connections = True):
	# Input
	inputs = Input(input_size)
	#encoding
	enc0 = EncodingLayer(inputs, kernel_size, number_of_kernels, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections, isInput = True)
	enc1 = EncodingLayer(enc0, kernel_size, number_of_kernels * 2, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections)
	enc2 = EncodingLayer(enc1, kernel_size, number_of_kernels * 4, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections)
	enc3 = EncodingLayer(enc2, kernel_size, number_of_kernels * 8, stride, max_pool, max_pool_size, batch_norm, residual_connections = residual_connections)
	#bottleneck without residual (might be without batch-norm)
	enc4 = EncodingLayer(enc3, kernel_size, number_of_kernels * 16, stride, False, max_pool_size, batch_norm, residual_connections = False)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec3 = DecodingLayer(enc4, enc3, 2, kernel_size, number_of_kernels * 8, batch_norm, residual_connections = residual_connections)
	dec2 = DecodingLayer(dec3, enc2, 2, kernel_size, number_of_kernels * 4, batch_norm, residual_connections = residual_connections)
	dec1 = DecodingLayer(dec2, enc1, 2, kernel_size, number_of_kernels * 2, batch_norm, residual_connections = residual_connections)
	dec0 = DecodingLayer(dec1, enc0, 2, kernel_size, number_of_kernels, batch_norm, residual_connections = residual_connections)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(dec0)
	model = Model(inputs, outputs)
	model.compile(optimizer = Adam(lr = 1e-4), loss = IOU_calc_loss, metrics = [dice_loss])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder5.png', show_shapes=True, show_layer_names=True)
	return model