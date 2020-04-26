# 5-layer UNet
from keras.models import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
from keras.layers import *

from models.layers import AtrousSpatialPyramidWaterFallPool, DecodingLayerAG_Res, AtrousSpatialPyramidPool, \
    EncodingLayer, DecodingLayer, DecodingLayerRes, EncodingLayerResAddOp, TripleDenseBottleneck
from models.losses import Loss, cross_and_dice_loss, weighted_cross_and_dice_loss, dice_score, \
    adjusted_weighted_bce_loss, weighted_bce_loss, FocalLoss, surface_loss, Active_Contour_Loss, binary_crossentropy, \
    dice_loss


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
    plot_model(model, to_file='UNet4.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet5_First5x5.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet4_res.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet5_res.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet5_res_First5x5.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet4_res_dense_aspp.png', show_shapes=True, show_layer_names=True)
    return model

###############3
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
    plot_model(model, to_file='UNet4_res_aspp.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet5_res_aspp.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet5_res_aspp_First5x5.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet4_res_aspp_AG.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet4_res_asppWF_AG.png', show_shapes=True, show_layer_names=True)
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
    plot_model(model, to_file='UNet4_res_asppWF.png', show_shapes=True, show_layer_names=True)
    return model