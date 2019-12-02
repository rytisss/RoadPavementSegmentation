# 5-layer UNet
from keras.models import *
from script.model.losses import *
from script.model.layers import *
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
    elif lossFunction == Loss.CROSSnDICE:
        model.compile(optimizer=Adam(lr=1e-3), loss=weighted_bce_dice_loss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTEDCROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=weighted_bce_loss, metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED60CROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=adjusted_weighted_bce_loss(0.6), metrics=[dice_score])
    elif lossFunction == Loss.WEIGHTED70CROSSENTROPY:
        model.compile(optimizer=Adam(lr=1e-3), loss=adjusted_weighted_bce_loss(0.7), metrics=[dice_score])
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
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, 5, stride, max_pool, max_pool_size, batch_norm)
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
    plot_model(model, to_file='AutoEncoderRes4shorcutAdditionToOpConcRes.png', show_shapes=True, show_layer_names=True)
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
