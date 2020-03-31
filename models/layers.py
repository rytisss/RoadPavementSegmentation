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
    concatenatedInput = concatenate([conv, skippedInput], axis=3)
    # shortcut
    shortcut = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')(concatenatedInput)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenatedInput)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    # add shortcut
    conv = Add()([conv, shortcut])

    conv = Activation('relu')(conv)
    output = conv
    return output

def DecodingLayerAG_Res(input,
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
    #add attention gate
    attention_gate = AttentionBlock(conv, skippedInput, kernels)
    #
    concatenatedInput = concatenate([conv, attention_gate], axis=3)
    # shortcut
    shortcut = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')(concatenatedInput)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenatedInput)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    # add shortcut
    conv = Add()([conv, shortcut])

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

def AttentionBlock(x, shortcut, filters):
    # theta_x(?,g_height,g_width,inter_channel)
    theta_x = Conv2D(filters, [1, 1], strides=[1, 1])(x)
    #batch norm
    # phi_g(?,g_height,g_width,inter_channel)
    phi_g = Conv2D(filters, [1, 1], strides=[1, 1])(shortcut)
    # batch norm
    # f(?,g_height,g_width,inter_channel)
    theta_x_phi_g = Add()([theta_x, phi_g])
    f = Activation('relu')(theta_x_phi_g)
    # psi_f(?,g_height,g_width,1)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    # rate(?,x_height,x_width)
    # att_x(?,x_height,x_width,x_channel)
    att_x = multiply([x, rate])
    return att_x

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
    if batch_norm == True:
        conv = BatchNormalization()(conv)

    # add shortcut
    conv = Add()([conv, shortcut])

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

def TripleDenseBottleneck(input,
                    kernels=8,
                    kernel_size=3):

    #make input dimension same before connecting to following layers
    input_1 = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                  kernel_initializer='he_normal')(input)
    input_1 = BatchNormalization()(input_1)
    input_1 = Activation('relu')(input_1)
    #First
    conv1 = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(input_1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)


    #input to second layer
    input_2 = concatenate([input_1, conv1])
    #1x1 possible to reduce features in entry
    input_2 = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                     kernel_initializer='he_normal')(input_2)
    input_2 = BatchNormalization()(input_2)
    input_2 = Activation('relu')(input_2)

    conv2 = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(input_2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)


    #input to third layer
    input_3 = concatenate([input_1, conv1, conv2])
    # 1x1 possible to reduce features in entry
    input_3 = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                   kernel_initializer='he_normal')(input_3)
    input_3 = BatchNormalization()(input_3)
    input_3 = Activation('relu')(input_3)

    conv3 = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                   kernel_initializer='he_normal')(input_3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    output = concatenate([input_1, conv1, conv2, conv3])

    #reduce number of parameters with 1x1
    output = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                  kernel_initializer='he_normal')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    return output

def AtrousSpatialPyramidPool(input,
                          kernels=8,
                          kernel_size=3):

    # dilate = 1
    dilate1 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=1, kernel_initializer='he_normal')(input)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)

    # dilate = 2
    dilate2 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=2, kernel_initializer='he_normal')(input)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)

    # dilate = 3
    dilate3 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=4, kernel_initializer='he_normal')(input)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)

    H, W, n_ch = input.shape.as_list()[1:]
    pool = AveragePooling2D(pool_size=(H, W))(input)
    pool = UpSampling2D((H, W), interpolation='bilinear')(pool)

    # pool
    #pool = AveragePooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(input)
    #averaged_pool = Conv2D(kernels, 1, strides=(1,1), padding='same', kernel_initializer='he_normal')(input)
    #conv = BatchNormalization()(conv)
    #conv = Activation('relu')(conv)

    output = concatenate([dilate1, dilate2, dilate3, pool])

    #perform parameters reduction with 1x1
    output = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                    kernel_initializer='he_normal')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    return output

def AtrousSpatialPyramidWaterFallPool(input,
                          kernels=8,
                          kernel_size=3):

    # dilate = 1
    dilate1 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=1, kernel_initializer='he_normal')(input)
    dilate1 = BatchNormalization()(dilate1)
    dilate1 = Activation('relu')(dilate1)

    # dilate = 2
    dilate2 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=2, kernel_initializer='he_normal')(dilate1)
    dilate2 = BatchNormalization()(dilate2)
    dilate2 = Activation('relu')(dilate2)

    # dilate = 3
    dilate3 = Conv2D(kernels, kernel_size, padding='same', dilation_rate=4, kernel_initializer='he_normal')(dilate2)
    dilate3 = BatchNormalization()(dilate3)
    dilate3 = Activation('relu')(dilate3)

    H, W, n_ch = input.shape.as_list()[1:]
    pool = AveragePooling2D(pool_size=(H, W))(input)
    pool = UpSampling2D((H, W), interpolation='bilinear')(pool)

    # pool
    #pool = AveragePooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(input)
    #averaged_pool = Conv2D(kernels, 1, strides=(1,1), padding='same', kernel_initializer='he_normal')(input)
    #conv = BatchNormalization()(conv)
    #conv = Activation('relu')(conv)

    output = concatenate([dilate1, dilate2, dilate3, pool])

    #perform parameters reduction with 1x1
    output = Conv2D(kernels, kernel_size=(1, 1), strides=1, padding='same',
                    kernel_initializer='he_normal')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    return output