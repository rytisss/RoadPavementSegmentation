from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *

from .losses import *
from .layers import *

def CompileModel(model, lossFunction, learning_rate = 1e-3):
    if lossFunction == Loss.DICE:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss, metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=binary_crossentropy, metrics=[dice_score])
    elif lossFunction == Loss.ACTIVECONTOURS:
        model.compile(optimizer=Adam(lr=learning_rate), loss=Active_Contour_Loss, metrics=[dice_score])
    elif lossFunction == Loss.SURFACEnDice:
        model.compile(optimizer=Adam(lr=learning_rate), loss=surface_loss, metrics=[dice_score])
    elif lossFunction == Loss.FOCALLOSS:
        model.compile(optimizer=Adam(lr=learning_rate), loss=FocalLoss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_bce_loss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED60CROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=adjusted_weighted_bce_loss(0.6), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED70CROSSENTROPY:
        model.compile(optimizer=Adam(lr=learning_rate), loss=adjusted_weighted_bce_loss(0.7), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY50DICE50:
        model.compile(optimizer=Adam(lr=learning_rate), loss=cross_and_dice_loss(0.5, 0.5), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY25DICE75:
        model.compile(optimizer=Adam(lr=learning_rate), loss=cross_and_dice_loss(0.25, 0.75), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY75DICE25:
        model.compile(optimizer=Adam(lr=learning_rate), loss=cross_and_dice_loss(0.75, 0.25), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY50DICE50:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_cross_and_dice_loss(0.5, 0.5), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY25DICE75:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_cross_and_dice_loss(0.25, 0.75), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY75DICE25:
        model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_cross_and_dice_loss(0.75, 0.25), metrics=[dice_score])
    return model

def UNet4(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4.png', show_shapes=True, show_layer_names=True)
    return model

def UNet5_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc4, enc4 = EncodingLayer(enc3, number_of_kernels * 16, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec3 = DecodingLayer(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
    dec2 = DecodingLayer(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                useLeakyReLU=True,
                LeakyReLU_alpha=0.1,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_First5x5_GroupConv(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                useLeakyReLU=True,
                LeakyReLU_alpha=0.1,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayerGroupConv(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingLayerGroupConv(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingLayerGroupConv(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    _, enc3 = EncodingLayerGroupConv(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerGroupConv(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingLayerGroupConv(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingLayerGroupConv(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

# First convolutional operation each downscale/upscale is deformed
def UNet4_First5x5_FirstDeformable(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                useLeakyReLU=True,
                LeakyReLU_alpha=0.1,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingFirstDeformableConv2DLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingFirstDeformableConv2DLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingFirstDeformableConv2DLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    _, enc3 = EncodingFirstDeformableConv2DLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingFirstDeformableConv2DLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingFirstDeformableConv2DLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingFirstDeformableConv2DLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

# Both convolutional operation each downscale/upscale layers are deformed
def UNet4_First5x5_BothDeformable(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                useLeakyReLU=True,
                LeakyReLU_alpha=0.1,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingBothDeformableConv2DLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingBothDeformableConv2DLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingBothDeformableConv2DLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    _, enc3 = EncodingBothDeformableConv2DLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingBothDeformableConv2DLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingBothDeformableConv2DLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingBothDeformableConv2DLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_Coordconv_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                useLeakyReLU=True,
                LeakyReLU_alpha=0.1,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingCoordConvLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingCoordConvLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingCoordConvLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    _, enc3 = EncodingCoordConvLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerCoordConv(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingLayerCoordConv(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingLayerCoordConv(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_Coordconv_First5x5_BothDeformable(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                useLeakyReLU=True,
                LeakyReLU_alpha=0.1,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingCoordConvLayerBothDeformable(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingCoordConvLayerBothDeformable(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingCoordConvLayerBothDeformable(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    _, enc3 = EncodingCoordConvLayerBothDeformable(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerCoordConvBothDeformable(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingLayerCoordConvBothDeformable(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingLayerCoordConvBothDeformable(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU=useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_First5x5_OctaveConv2D(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                alpha = 0.5,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    input_high = inputs
    input_low = AvgPool2D((2,2))(inputs)
    opposite_connection_high_0, opposite_connection_low_0, high_0, low_0 = EncodingOctaveConv2D(input_high,
                                                                                                input_low,
                                                                                                number_of_kernels,
                                                                                                kernel_size=5,
                                                                                                max_pool=True,
                                                                                                max_pool_size=2,
                                                                                                batch_norm=True,
                                                                                                alpha=alpha)
    opposite_connection_high_1, opposite_connection_low_1, high_1, low_1 = EncodingOctaveConv2D(high_0,
                                                                                                low_0,
                                                                                                number_of_kernels * 2,
                                                                                                kernel_size=3,
                                                                                                max_pool=True,
                                                                                                max_pool_size=2,
                                                                                                batch_norm=True,
                                                                                                alpha=alpha)
    opposite_connection_high_2, opposite_connection_low_2, high_2, low_2 = EncodingOctaveConv2D(high_1,
                                                                                                low_1,
                                                                                                number_of_kernels * 4,
                                                                                                kernel_size=3,
                                                                                                max_pool=True,
                                                                                                max_pool_size=2,
                                                                                                batch_norm=True,
                                                                                                alpha=alpha)
    _, _, high_3, low_3 = EncodingOctaveConv2D(high_2,
                                                low_2,
                                                number_of_kernels * 8,
                                                kernel_size=3,
                                                max_pool=False,
                                                max_pool_size=2,
                                                batch_norm=True,
                                                alpha=alpha)

    dec_high_2, dec_low_2 = DecodingOctaveConv2D(high_3,
                                                 low_3,
                                                 opposite_connection_high_2,
                                                 opposite_connection_low_2,
                                                 number_of_kernels * 4,
                                                 kernel_size=3,
                                                 batch_norm=True,
                                                 alpha=alpha)
    dec_high_1, dec_low_1 = DecodingOctaveConv2D(dec_high_2,
                                                 dec_low_2,
                                                 opposite_connection_high_1,
                                                 opposite_connection_low_1,
                                                 number_of_kernels * 2,
                                                 kernel_size=3,
                                                 batch_norm=True,
                                                 alpha=alpha)
    dec_high_0, dec_low_0 = DecodingOctaveConv2D(dec_high_1,
                                                 dec_low_1,
                                                 opposite_connection_high_0,
                                                 opposite_connection_low_0,
                                                 number_of_kernels,
                                                 kernel_size=3,
                                                 batch_norm=True,
                                                 alpha=alpha)

    channels = int(inputs.shape[-1])
    h2h = Conv2D(channels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same')(dec_high_0) # to make number of filter1 1
    l2h = Conv2DTranspose(channels, kernel_size=(kernel_size, kernel_size), strides=(2, 2), padding='same')(dec_low_0) # to upscale and make number of filters 1
    x = Add()([h2h, l2h])

    x = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(x)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    # plot_model(model, to_file='UNet4_res.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_res(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size,
                                               batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_assp(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_assp.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_asspWF(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    assp = AtrousSpatialPyramidWaterFallPool(enc3, number_of_kernels * 8, kernel_size)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_asspWF.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_assp_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_assp_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_asspWF_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    assp = AtrousSpatialPyramidWaterFallPool(enc3, number_of_kernels * 8, kernel_size)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_asspWF_First5x5.png', show_shapes=True, show_layer_names=True)
    return model


def UNet4_assp_AG_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer_AG(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer_AG(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_assp_AG_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_asspWF_AG_First5x5(pretrained_weights=None,
                 input_size=(320, 320, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY,
                 learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    _, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    assp = AtrousSpatialPyramidWaterFallPool(enc3, number_of_kernels * 8, kernel_size)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer_AG(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer_AG(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_asspWF_AG_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet5_res(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
              learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool,
                                               max_pool_size,
                                               batch_norm)
    _, enc4 = EncodingLayerResAddOp(enc3, number_of_kernels * 16, kernel_size, stride, False,
                                               max_pool_size,
                                               batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec3 = DecodingLayerRes(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
    dec2 = DecodingLayerRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_res.png', show_shapes=True, show_layer_names=True)
    return model

def UNet5_res_First5x5(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
              learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool,
                                               max_pool_size,
                                               batch_norm)
    _, enc4 = EncodingLayerResAddOp(enc3, number_of_kernels * 16, kernel_size, stride, False,
                                               max_pool_size,
                                               batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec3 = DecodingLayerRes(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
    dec2 = DecodingLayerRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_res_First5x5.png', show_shapes=True, show_layer_names=True)
    return model


def UNet4_res_dense_aspp(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    dense_block = TripleDenseBottleneck(enc2, number_of_kernels * 8, kernel_size)
    assp = AtrousSpatialPyramidPool(dense_block, number_of_kernels * 8, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_dense_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############
def UNet4_res_aspp(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    _, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############
def UNet4_res_aspp_First5x5(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                useLeakyReLU=True,
                                LeakyReLU_alpha=0.1,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    _, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############
def UNet4_res_aspp_First5x5_FirstDeformable(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                useLeakyReLU=True,
                                LeakyReLU_alpha=0.1,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingFirstDeformableConv2DLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingFirstDeformableConv2DLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingFirstDeformableConv2DLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    _, enc3 = EncodingFirstDeformableConv2DLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingFirstDeformableConv2DLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingFirstDeformableConv2DLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingFirstDeformableConv2DLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############
def UNet4_res_aspp_First5x5_BothDeformable(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                useLeakyReLU=True,
                                LeakyReLU_alpha=0.1,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingBothDeformableConv2DLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingBothDeformableConv2DLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingBothDeformableConv2DLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    _, enc3 = EncodingBothDeformableConv2DLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingBothDeformableConv2DLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec1 = DecodingBothDeformableConv2DLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)
    dec0 = DecodingBothDeformableConv2DLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU, leakyReLU_alpha=LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############
def UNet4_res_aspp_First5x5_CoordConv(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                useLeakyReLU=True,
                                LeakyReLU_alpha=0.1,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingCoordConvLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU, LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingCoordConvLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingCoordConvLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)

    _, enc3 = EncodingCoordConvLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    assp = AtrousSpatialPyramidPoolCoordConv(enc3, number_of_kernels * 8, kernel_size, useLeakyReLU, LeakyReLU_alpha)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingCoordConvLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    dec1 = DecodingCoordConvLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    dec0 = DecodingCoordConvLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = LeakyReLU(alpha=LeakyReLU_alpha)(dec0) if useLeakyReLU else Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model


def UNet4_res_aspp_First5x5_CoordConvBothDeformable(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                useLeakyReLU=True,
                                LeakyReLU_alpha=0.1,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingCoordConvLayerBothDeformable(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm, useLeakyReLU, LeakyReLU_alpha)
    oppositeEnc1, enc1 = EncodingCoordConvLayerResAddOpBothDeformable(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    oppositeEnc2, enc2 = EncodingCoordConvLayerResAddOpBothDeformable(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)

    _, enc3 = EncodingCoordConvLayerResAddOpBothDeformable(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    assp = AtrousSpatialPyramidPoolCoordConv(enc3, number_of_kernels * 8, kernel_size, useLeakyReLU, LeakyReLU_alpha)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingCoordConvLayerResBothDeformable(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    dec1 = DecodingCoordConvLayerResBothDeformable(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)
    dec0 = DecodingCoordConvLayerResBothDeformable(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm, useLeakyReLU, LeakyReLU_alpha)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)

    dec0 = LeakyReLU(alpha=LeakyReLU_alpha)(dec0) if useLeakyReLU else Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############3
def UNet5_res_aspp(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                   learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    _, enc4 = EncodingLayerResAddOp(enc3, number_of_kernels * 16, kernel_size, stride, False,
                                               max_pool_size, batch_norm)
    assp = AtrousSpatialPyramidPool(enc4, number_of_kernels * 16, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec3 = DecodingLayerRes(assp, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
    dec2 = DecodingLayerRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_res_aspp.png', show_shapes=True, show_layer_names=True)
    return model

def UNet5_res_aspp_First5x5(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                   learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    _, enc4 = EncodingLayerResAddOp(enc3, number_of_kernels * 16, kernel_size, stride, False,
                                               max_pool_size, batch_norm)
    assp = AtrousSpatialPyramidPool(enc4, number_of_kernels * 16, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec3 = DecodingLayerRes(assp, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
    dec2 = DecodingLayerRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet5_res_aspp_First5x5.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_res_aspp_AG(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    _, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm)
    assp = AtrousSpatialPyramidPool(enc3, number_of_kernels * 8, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerAG_Res(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerAG_Res(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_aspp_AG.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_res_asppWF_AG(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    _, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm)
    assp = AtrousSpatialPyramidWaterFallPool(enc3, number_of_kernels * 8, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerAG_Res(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerAG_Res(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_asppWF_AG.png', show_shapes=True, show_layer_names=True)
    return model

def UNet4_res_asppWF(pretrained_weights=None,
                                input_size=(320, 320, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY,
                                learning_rate = 1e-3):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)

    _, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False,
                                               max_pool_size, batch_norm)
    assp = AtrousSpatialPyramidWaterFallPool(enc3, number_of_kernels * 8, kernel_size)

    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerRes(assp, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function, learning_rate)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='UNet4_res_asppWF.png', show_shapes=True, show_layer_names=True)
    return model