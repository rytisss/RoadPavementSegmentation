# 5-layer UNet
from keras.models import *
from losses import *
from layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
"""
WEIGHTED60CROSSENTROPY = 7,
    WEIGHTED70CROSSENTROPY = 8,
    CROSSENTROPY50DICE50 = 9,
    CROSSENTROPY25DICE75 = 10,
    CROSSENTROPY25DICE75 = 11
    """


def CompileModel(model, lossFunction):
    if lossFunction == Loss.DICE:
        model.compile(optimizer=Adam(lr=1e-3), loss=dice_loss, metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[dice_score])
    elif lossFunction == Loss.ACTIVECONTOURS:
        model.compile(optimizer=Adam(lr=1e-3), loss=Active_Contour_Loss, metrics=[dice_score])
    elif lossFunction == Loss.SURFACEnDice:
        model.compile(optimizer=Adam(lr=1e-3), loss=surface_loss, metrics=[dice_score])
    elif lossFunction == Loss.FOCALLOSS:
        model.compile(optimizer=Adam(lr=5e-3), loss=FocalLoss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=weighted_bce_loss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED60CROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=adjusted_weighted_bce_loss(0.6), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED70CROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=adjusted_weighted_bce_loss(0.7), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY50DICE50:
        model.compile(optimizer=Adam(lr=1e-3), loss=cross_and_dice_loss(0.5, 0.5), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY25DICE75:
        model.compile(optimizer=Adam(lr=1e-3), loss=cross_and_dice_loss(0.25, 0.75), metrics=[dice_score])
    elif lossFunction == Loss.CROSSENTROPY75DICE25:
        model.compile(optimizer=Adam(lr=1e-3), loss=cross_and_dice_loss(0.75, 0.25), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY50DICE50:
        model.compile(optimizer=Adam(lr=1e-3), loss=weighted_cross_and_dice_loss(0.5, 0.5), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY25DICE75:
        model.compile(optimizer=Adam(lr=1e-3), loss=weighted_cross_and_dice_loss(0.25, 0.75), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY75DICE25:
        model.compile(optimizer=Adam(lr=1e-3), loss=weighted_cross_and_dice_loss(0.75, 0.25), metrics=[dice_score])
    return model

def AutoEncoder5(pretrained_weights=None,
                 input_size=(320, 480, 1),
                 kernel_size=3,
                 number_of_kernels=16,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
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
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder5.png', show_shapes=True, show_layer_names=True)
    return model


# 5-layer UNet with residual connection, opposite connectio with residual connections addition
def AutoEncoder5ResAddOp(pretrained_weights=None,
                         input_size=(320, 480, 1),
                         kernel_size=3,
                         number_of_kernels=16,
                         stride=1,
                         max_pool=True,
                         max_pool_size=2,
                         batch_norm=True,
                         loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc4, enc4 = EncodingLayer(enc3, number_of_kernels * 16, kernel_size, stride, False, max_pool_size,
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
    # CompileModel with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoderRes5shorcutAdditionToOp.png', show_shapes=True, show_layer_names=True)
    return model


# 5-layer UNet with residual connection, opposite connection with residual connections addition.
# Concat operation into residual connection in decoding
def AutoEncoder5ResAddOpConcDec(pretrained_weights=None,
                                input_size=(320, 480, 1),
                                kernel_size=3,
                                number_of_kernels=16,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc4, enc4 = EncodingLayer(enc3, number_of_kernels * 16, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec3 = DecodingLayerConcRes(enc4, oppositeEnc3, 2, number_of_kernels * 8, kernel_size, batch_norm)
    dec2 = DecodingLayerConcRes(dec3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoderRes5shorcutAdditionToOpConcRes.png', show_shapes=True, show_layer_names=True)
    return model


###4 layer
# 4-layer UNet
def AutoEncoder4(pretrained_weights=None,
                 input_size=(320, 480, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
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
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder4.png', show_shapes=True, show_layer_names=True)
    return model


def AutoEncoder4_5x5(pretrained_weights=None,
                     input_size=(320, 480, 1),
                     kernel_size=3,
                     number_of_kernels=32,
                     stride=1,
                     max_pool=True,
                     max_pool_size=2,
                     batch_norm=True,
                     loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size, batch_norm, True)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
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
    CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder4_5x5.png', show_shapes=True, show_layer_names=True)
    return model


# 4-layer UNet VGG16
def AutoEncoder4VGG16(pretrained_weights=None,
                      input_size=(320, 480, 1),
                      kernel_size=3,
                      number_of_kernels=32,
                      stride=1,
                      max_pool=True,
                      max_pool_size=2,
                      batch_norm=True,
                      loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayerTripple(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                              batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerTripple(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
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
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder4VGG16.png', show_shapes=True, show_layer_names=True)
    return model


def AutoEncoder4VGG16_5x5(pretrained_weights=None,
                          input_size=(320, 480, 1),
                          kernel_size=3,
                          number_of_kernels=32,
                          stride=1,
                          max_pool=True,
                          max_pool_size=2,
                          batch_norm=True,
                          loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size, batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayerTripple(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                              batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerTripple(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
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
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder4VGG16_5x5.png', show_shapes=True, show_layer_names=True)
    return model


# 4-layer UNet VGG16
def AutoEncoder4VGG19(pretrained_weights=None,
                      input_size=(320, 480, 1),
                      kernel_size=3,
                      number_of_kernels=32,
                      stride=1,
                      max_pool=True,
                      max_pool_size=2,
                      batch_norm=True,
                      loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayerQuad(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                           batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerQuad(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                           batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm == True:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder4VGG19.png', show_shapes=True, show_layer_names=True)
    return model


# 4-layer UNet with residual connection, opposite connectio with residual connections addition
def AutoEncoder4ResAddOp(pretrained_weights=None,
                         input_size=(320, 480, 1),
                         kernel_size=3,
                         number_of_kernels=32,
                         stride=1,
                         max_pool=True,
                         max_pool_size=2,
                         batch_norm=True,
                         loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
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
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOp.png', show_shapes=True, show_layer_names=True)
    return model


# 4-layer UNet with residual connection, opposite connectio with residual connections addition
def AutoEncoder4ResAddOpFirstEx(pretrained_weights=None,
                                input_size=(320, 480, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
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
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpFirstEx.png', show_shapes=True, show_layer_names=True)
    return model


# 4-layer UNet with residual connection, opposite connection with residual connections addition.
# Concat operation into residual connection in decoding
def AutoEncoder4ResAddOpConcDec(pretrained_weights=None,
                                input_size=(320, 480, 1),
                                kernel_size=3,
                                number_of_kernels=32,
                                stride=1,
                                max_pool=True,
                                max_pool_size=2,
                                batch_norm=True,
                                loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayerResAddOp(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                               batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                               batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerConcRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpConcRes.png', show_shapes=True, show_layer_names=True)
    return model


def AutoEncoder4ResAddOpConcDecFirstEx(pretrained_weights=None,
                                       input_size=(320, 480, 1),
                                       kernel_size=3,
                                       number_of_kernels=32,
                                       stride=1,
                                       max_pool=True,
                                       max_pool_size=2,
                                       batch_norm=True,
                                       loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                               batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerConcRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpConcResFirstEx.png', show_shapes=True,
               show_layer_names=True)
    return model


def AutoEncoder4ResAddOpConcDecFirstEx_5x5(pretrained_weights=None,
                                           input_size=(320, 480, 1),
                                           kernel_size=3,
                                           number_of_kernels=32,
                                           stride=1,
                                           max_pool=True,
                                           max_pool_size=2,
                                           batch_norm=True,
                                           loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size, batch_norm)
    oppositeEnc1, enc1 = EncodingLayerResAddOp(enc0, number_of_kernels * 2, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    oppositeEnc2, enc2 = EncodingLayerResAddOp(enc1, number_of_kernels * 4, kernel_size, stride, max_pool,
                                               max_pool_size, batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayerResAddOp(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                               batch_norm)
    # decoding
    # Upsample rate needs to be same as downsampling! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayerConcRes(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayerConcRes(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayerConcRes(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='AutoEncoder4ResAddOpConcDecFirstEx_5x5.png', show_shapes=True, show_layer_names=True)
    return model


def BCDU_net_D3(input_size=(320, 480, 1)):
    H = input_size[0]
    W = input_size[1]
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4_1)
    conv4_2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2, drop4_1], axis=3)
    conv4_3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_dense)
    conv4_3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(H / 4), np.int32(W / 4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(H / 4), np.int32(W / 4), 256))(up6)
    merge6 = concatenate([x1, x2], axis=1)
    merge6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(H / 2), np.int32(W / 2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(H / 2), np.int32(W / 2), 128))(up7)
    merge7 = concatenate([x1, x2], axis=1)
    merge7 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)

    x1 = Reshape(target_shape=(1, H, W, 64))(conv1)
    x2 = Reshape(target_shape=(1, H, W, 64))(up8)
    merge8 = concatenate([x1, x2], axis=1)
    merge8 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge8)

    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)

    model = Model(input=inputs, output=conv9)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model
