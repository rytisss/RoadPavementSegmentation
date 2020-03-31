from keras.layers import *


def DecodingLayer(input,
                  skippedInput,
                  upSampleSize=2,
                  kernels=8,
                  kernel_size=3,
                  batch_norm=True):
    conv = Conv2D(kernels, kernel_size=(2, 2), padding='same', kernel_initializer='he_normal')(
        UpSampling2D((upSampleSize, upSampleSize))(input))
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    concatenatedInput = concatenate([conv, skippedInput], axis=3)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenatedInput)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    output = conv
    return output


def DecodingLayerRes(input,
                     skippedInput,
                     upSampleSize=2,
                     kernels=8,
                     kernel_size=3,
                     batch_norm=True):
    conv = Conv2D(kernels, kernel_size=(2, 2), padding='same', kernel_initializer='he_normal')(
        UpSampling2D((upSampleSize, upSampleSize))(input))
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    # shortcut
    shortcut = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')(conv)
    concatenatedInput = concatenate([conv, skippedInput], axis=3)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenatedInput)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    # add shortcut
    conv = Add()([conv, shortcut])
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    output = conv
    return output


# shortcut with concated layer (dimension will be reduced in half in 1x1 convolution shortcut)
def DecodingLayerConcRes(input,
                         skippedInput,
                         upSampleSize=2,
                         kernels=8,
                         kernel_size=3,
                         batch_norm=True):
    conv = Conv2D(kernels, kernel_size=(2, 2), padding='same', kernel_initializer='he_normal')(
        UpSampling2D((upSampleSize, upSampleSize))(input))
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    concatenatedInput = concatenate([conv, skippedInput], axis=3)
    # shortcut with concat
    shortcut = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')(
        concatenatedInput)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenatedInput)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    # add shortcut
    conv = Add()([conv, shortcut])
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    output = conv
    return output


def EncodingLayer(input,
                  kernels=8,
                  kernel_size=3,
                  stride=1,
                  max_pool=True,
                  max_pool_size=2,
                  batch_norm=True,
                  isInput=False):
    # Double convolution according to U-Net structure
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same',
                  kernel_initializer='he_normal')(input)
    # Batch-normalization on demand
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
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
    # in next step this output needs to be activated
    return oppositeConnection, output


def EncodingLayerTripple(input,
                         kernels=8,
                         kernel_size=3,
                         stride=1,
                         max_pool=True,
                         max_pool_size=2,
                         batch_norm=True,
                         isInput=False):
    # Double convolution according to U-Net structure
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same',
                  kernel_initializer='he_normal')(input)
    # Batch-normalization on demand
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
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
    # in next step this output needs to be activated
    return oppositeConnection, output


def EncodingLayerQuad(input,
                      kernels=8,
                      kernel_size=3,
                      stride=1,
                      max_pool=True,
                      max_pool_size=2,
                      batch_norm=True,
                      isInput=False):
    # Double convolution according to U-Net structure
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same',
                  kernel_initializer='he_normal')(input)
    # Batch-normalization on demand
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
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
    # in next step this output needs to be activated
    return oppositeConnection, output


def EncodingLayerResAddOp(input,
                          kernels=8,
                          kernel_size=3,
                          stride=1,
                          max_pool=True,
                          max_pool_size=2,
                          batch_norm=True,
                          isInput=False):
    # Double convolution according to U-Net structure
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same',
                  kernel_initializer='he_normal')(input)

    # calculate how many times
    downscale = stride
    shortcut = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')(input)
    if downscale != 1:
        shortcut = MaxPooling2D(pool_size=(downscale, downscale))(shortcut)

    # Batch-normalization on demand
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)

    # add shortcut
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
    # in next step this output needs to be activated
    return oppositeConnection, output
