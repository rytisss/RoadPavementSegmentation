import numpy as np 
import os

#import skimage.io as io
#import skimage.transform as trans
import numpy as np
from enum import Enum
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as keras

from keras.utils.vis_utils import plot_model
from scipy.ndimage import distance_transform_edt as distance

class Loss(Enum):
	CROSSENTROPY = 0,
	DICE = 1,
	ACTIVECONTOURS = 2,
	SURFACEnDice = 3
#------> 
alpha = K.variable(1, dtype='float32')

class AlphaScheduler(Callback):
 def on_epoch_end(self, epoch, logs=None):
  alpha = alpha - 0.01

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled) 

def SurficenDiceLoss(y_true, y_pred):
	alpha_ = 0.5
	dice = IOU_calc_loss(y_true, y_pred) * alpha_
	surface = surface_loss(y_true, y_pred) * (1.0 - alpha_)
	return dice + surface

def dice_loss(y_true, y_pred):
	smooth = 1e-6
	y_true_f = keras.flatten(y_true)
	y_pred_f = keras.flatten(y_pred)
	intersection = keras.sum(y_true_f * y_pred_f)
	answer = (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
	return answer
	
def IOU_calc_loss(y_true, y_pred):
	return 1 - dice_loss(y_true, y_pred)

#non-working
def Active_Contour_Loss(y_true, y_pred): 

	#y_pred = K.cast(y_pred, dtype = 'float64')

	"""
	lenth term
	"""

	x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
	y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

	delta_x = x[:,:,1:,:-2]**2
	delta_y = y[:,:,:-2,1:]**2
	delta_u = K.abs(delta_x + delta_y) 

	epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
	w = 1
	lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

	"""
	region term
	"""

	C_1 = np.ones((480, 320))
	C_2 = np.zeros((480, 320))

	region_in = K.abs(K.sum( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = K.abs(K.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 1 # lambda parameter could be various.
	
	loss =  lenth + lambdaP * (region_in + region_out) 

	return loss

def Active_Contour_loss_minimization(y_true, y_pred):
	return Active_Contour_Loss(y_true, y_pred)

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
			kernels = 8,
			kernel_size = 3,
			stride = 1,
			max_pool = True,
			max_pool_size = 2,
			batch_norm = True,
			isInput = False):
	
	# Double convolution according to U-Net structure
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = stride, padding = 'same', kernel_initializer = 'he_normal')(input)
	# Batch-normalization on demand
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	# Max-pool on demand
	if max_pool == True:
		oppositeConnection = conv
		output = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
	else:
		oppositeConnection = conv
		output = conv
	#in next step this output needs to be activated
	return oppositeConnection, output

def EncodingLayerTripple(input,
			kernels = 8,
			kernel_size = 3,
			stride = 1,
			max_pool = True,
			max_pool_size = 2,
			batch_norm = True,
			isInput = False):
	
	# Double convolution according to U-Net structure
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = stride, padding = 'same', kernel_initializer = 'he_normal')(input)
	# Batch-normalization on demand
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	# Max-pool on demand
	if max_pool == True:
		oppositeConnection = conv
		output = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
	else:
		oppositeConnection = conv
		output = conv
	#in next step this output needs to be activated
	return oppositeConnection, output

def EncodingLayerQuad(input,
			kernels = 8,
			kernel_size = 3,
			stride = 1,
			max_pool = True,
			max_pool_size = 2,
			batch_norm = True,
			isInput = False):
	
	# Double convolution according to U-Net structure
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = stride, padding = 'same', kernel_initializer = 'he_normal')(input)
	# Batch-normalization on demand
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	
	# Max-pool on demand
	if max_pool == True:
		oppositeConnection = conv
		output = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
	else:
		oppositeConnection = conv
		output = conv
	#in next step this output needs to be activated
	return oppositeConnection, output

def EncodingLayerResAddOp(input,
			kernels = 8,
			kernel_size = 3,
			stride = 1,
			max_pool = True,
			max_pool_size = 2,
			batch_norm = True,
			isInput = False):
	
	# Double convolution according to U-Net structure
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = stride, padding = 'same', kernel_initializer = 'he_normal')(input)

	# calculate how many times
	downscale = stride
	shortcut = Conv2D(kernels, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(input)
	if downscale != 1:
		shortcut = MaxPooling2D(pool_size=(downscale, downscale))(shortcut)

	# Batch-normalization on demand
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)

	#add shortcut
	conv = Add()([conv, shortcut])

	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	# Max-pool on demand
	if max_pool == True:
		oppositeConnection = conv
		output = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
	else:
		oppositeConnection = conv
		output = conv
	#in next step this output needs to be activated
	return oppositeConnection, output

"""
def EncodingLayer(input,
				kernel_size = 3,
				kernels = 8,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				residual_connections = False,
				isInput = False):
	
	#if batch_norm == True and isInput == False:
	#	shortcut = BatchNormalization()(shortcut)
	# do not make batch-normalization on the first layer/neural network input, because data here is already normalized!
	if batch_norm == True and isInput == False:
		input = BatchNormalization()(input)
	# do not activate in first layer
	if isInput == False:
		input = Activation('relu')(input)
	if residual_connections == True:
		# calculate how many times
		downscale = stride
		if max_pool == True:
			downscale *= max_pool_size
		shortcut = Conv2D(kernels, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(input)
		if max_pool == True:
			shortcut = MaxPooling2D(pool_size=(downscale, downscale))(shortcut) 
	# Double convolution according to U-Net structure
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = stride, padding = 'same', kernel_initializer = 'he_normal')(input)
	# Batch-normalization on demand
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	# Max-pool on demand
	if max_pool == True:
		conv = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
	# in case we are using residual connection, add shortcut
	if residual_connections == True:
		conv = Add()([conv, shortcut])
	#in next step this output needs to be activated
	output = conv
	return output

"""
def DecodingLayer(input,
				skippedInput,
				upSampleSize = 2,
				kernels = 8,
				kernel_size = 3,
				batch_norm = True):
	conv = Conv2D(kernels, kernel_size = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D((upSampleSize, upSampleSize))(input))
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	concatenatedInput = concatenate([conv, skippedInput], axis = 3)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(concatenatedInput)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)

	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	output = conv
	return output

def DecodingLayerRes(input,
				skippedInput,
				upSampleSize = 2,
				kernels = 8,
				kernel_size = 3,
				batch_norm = True):
	conv = Conv2D(kernels, kernel_size = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D((upSampleSize, upSampleSize))(input))

	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	#shortcut
	shortcut = Conv2D(kernels, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)

	concatenatedInput = concatenate([conv, skippedInput], axis = 3)
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(concatenatedInput)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)

	#add shortcut
	conv = Add()([conv, shortcut])

	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	output = conv
	return output

#shortcut with concated layer (dimension will be reduced in half in 1x1 convolution shortcut)
def DecodingLayerConcRes(input,
				skippedInput,
				upSampleSize = 2,
				kernels = 8,
				kernel_size = 3,
				batch_norm = True):
	conv = Conv2D(kernels, kernel_size = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D((upSampleSize, upSampleSize))(input))

	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	concatenatedInput = concatenate([conv, skippedInput], axis = 3)

	#shortcut with concat
	shortcut = Conv2D(kernels, kernel_size = (1, 1), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(concatenatedInput)

	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(concatenatedInput)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)

	#add shortcut
	conv = Add()([conv, shortcut])

	if batch_norm == True:
		conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)

	output = conv
	return output
	
"""
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
	conv = Conv2D(kernels, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv)
	if batch_norm == True:
		conv = BatchNormalization()(conv)
	if residual_connections == True:
		conv = Add()([conv, shortcut])
	output = conv
	return output
"""

#5-layer UNet
def AutoEncoder5(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 16,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc4, enc4 = EncodingLayer(enc3, number_of_kernels * 16, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec3 = DecodingLayer(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
	dec2 = DecodingLayer(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)

	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder5.png', show_shapes=True, show_layer_names=True)
	return model

#5-layer UNet with residual connection, opposite connectio with residual connections addition
def AutoEncoder5ResAddOp(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 16,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc4, enc4 = EncodingLayer(enc3, number_of_kernels * 16, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec3 = DecodingLayerRes(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
	dec2 = DecodingLayerRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes5shorcutAdditionToOp.png', show_shapes=True, show_layer_names=True)
	return model

#5-layer UNet with residual connection, opposite connection with residual connections addition. 
#Concat operation into residual connection in decoding
def AutoEncoder5ResAddOpConcDec(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 16,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc4, enc4 = EncodingLayer(enc3, number_of_kernels * 16, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec3 = DecodingLayerConcRes(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
	dec2 = DecodingLayerConcRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes5shorcutAdditionToOpConcRes.png', show_shapes=True, show_layer_names=True)
	return model


###4 layer
#4-layer UNet
def AutoEncoder4(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)

	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder4.png', show_shapes=True, show_layer_names=True)
	return model

def AutoEncoder4_5x5(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)

	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	elif (loss_function == Loss.ACTIVECONTOURS):
		model.compile(optimizer = Adam(lr = 1e-3), loss = Active_Contour_loss_minimization, metrics = [Active_Contour_Loss])
	elif (loss_function == Loss.SURFACEnDice):
		model.compile(optimizer = Adam(lr = 1e-3), loss = SurficenDiceLoss, metrics = [dice_loss])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder4_5x5.png', show_shapes=True, show_layer_names=True)
	return model
	

#4-layer UNet VGG16
def AutoEncoder4VGG16(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerTripple(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerTripple(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)

	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder4VGG16.png', show_shapes=True, show_layer_names=True)
	return model

def AutoEncoder4VGG16_5x5(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerTripple(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerTripple(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)

	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder4VGG16_5x5.png', show_shapes=True, show_layer_names=True)
	return model
	
#4-layer UNet VGG16
def AutoEncoder4VGG19(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerQuad(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerQuad(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)

	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder4VGG19.png', show_shapes=True, show_layer_names=True)
	return model

#4-layer UNet with residual connection, opposite connectio with residual connections addition
def AutoEncoder4ResAddOp(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayerRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOp.png', show_shapes=True, show_layer_names=True)
	return model

#4-layer UNet with residual connection, opposite connectio with residual connections addition
def AutoEncoder4ResAddOpFirstEx(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayerRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpFirstEx.png', show_shapes=True, show_layer_names=True)
	return model

#4-layer UNet with residual connection, opposite connection with residual connections addition. 
#Concat operation into residual connection in decoding
def AutoEncoder4ResAddOpConcDec(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayerConcRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpConcRes.png', show_shapes=True, show_layer_names=True)
	return model

def AutoEncoder4ResAddOpConcDecFirstEx(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayerConcRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpConcResFirstEx.png', show_shapes=True, show_layer_names=True)
	return model

def AutoEncoder4ResAddOpConcDecFirstEx_5x5(pretrained_weights = None,
				input_size = (320,480,1),
				kernel_size = 3,
				number_of_kernels = 32,
				stride = 1,
				max_pool = True,
				max_pool_size = 2,
				batch_norm = True,
				loss_function = Loss.CROSSENTROPY):
	# Input
	inputs = Input(input_size)
	#encoding
	oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5,  stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size, batch_norm)
	#bottleneck without residual (might be without batch-norm)
	#opposite connection is equal to enc4
	oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size, batch_norm)
	#decoding
	#Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
	dec2 = DecodingLayerConcRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size,  batch_norm)
	dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
	dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size,  batch_norm)
	
	dec0 = Conv2D(2, kernel_size = (kernel_size, kernel_size), strides = 1, padding = 'same', kernel_initializer = 'he_normal')(dec0)
	if batch_norm == True:
		dec0 = BatchNormalization()(dec0)
	dec0 = Activation('relu')(dec0)

	outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer = 'glorot_normal')(dec0)
	model = Model(inputs, outputs)
	if (loss_function == Loss.DICE):
		model.compile(optimizer = Adam(lr = 1e-3), loss = IOU_calc_loss, metrics = [dice_loss])
	elif (loss_function == Loss.CROSSENTROPY):
		model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
	# Load trained weights if they are passed here
	if (pretrained_weights):
		model.load_weights(pretrained_weights)
	plot_model(model, to_file='AutoEncoder4ResAddOpConcDecFirstEx_5x5.png', show_shapes=True, show_layer_names=True)
	return model