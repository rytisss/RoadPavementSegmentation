from enum import Enum
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import keras as K
import tensorflow as tf


# Define all possible losses
class Loss(Enum):
    CROSSENTROPY = 0,
    DICE = 1,
    ACTIVECONTOURS = 2,
    SURFACEnDice = 3,
    FOCALLOSS = 4,
    CROSSnDICE = 5,
    WEIGHTEDCROSSENTROPY = 6,
    WEIGHTED60CROSSENTROPY = 7,
    WEIGHTED70CROSSENTROPY = 8,
    CROSSENTROPY50DICE50 = 9,
    CROSSENTROPY25DICE75 = 10,
    CROSSENTROPY75DICE25 = 11


alpha = K.backend.variable(1.0, dtype='float32')

"""
Scheduling example

alpha = K.variable(1.0, dtype='float32')
class AlphaScheduler(Callback):
 def on_epoch_end(self, epoch, logs=None):
  alpha_ = K.get_value(alpha)
  alpha_ -= 0.01
  if alpha_ < 0.1:
   alpha_ = 0.1
  K.set_value(alpha, alpha_)
  print(alpha_)
"""


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

#get weight matrix for tensor (image or images stack)
def get_weight_matrix(y_true):
    y_true = K.backend.cast(y_true, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    #averaged_mask = K.pool2d(
    #    y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    #border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    # basically finds label, (non-black) points in tensor
    labelmatrix = K.backend.cast(K.backend.greater(y_true, 0.5), 'float32')
    weight = K.backend.ones_like(y_true)
    w0 = K.backend.sum(weight)
    weight += labelmatrix
    w1 = K.backend.sum(weight)
    weight *= (w0 / w1)
    return weight

def binary_crossentropy(y_true, y_pred):
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = y_true, y_pred = y_pred, from_logits = False))
    return loss

#get weight matrix for tensor (image or images stack)
def get_edge_matrix(y_true, min_kernel_overlay = 0.5, max_kernel_overlay = 0.8):
    y_true = K.backend.cast(y_true, 'float32')
    #if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.backend.pool2d(
        y_true, pool_size=(3, 3), strides=(1, 1), padding='same', pool_mode='avg')
    edge = K.backend.cast(K.backend.greater(averaged_mask, min_kernel_overlay), 'float32') * K.backend.cast(K.backend.less(averaged_mask, max_kernel_overlay), 'float32')
    #take everything that is the label only (not outside)
    edge *= y_true
    return edge

def get_weight_matrix_with_reduced_edges(y_true, max_kernel_overlay = 0.5):
    edge = get_edge_matrix(y_true, 0.1, max_kernel_overlay)
    # take the edges only in the label (not outside)
    edge *= y_true
    label_without_edge = y_true - edge
    return label_without_edge

def adjusted_weighted_bce_loss(max_kernel_overlay = 0.8):
    def adjusted_weighted_bce_loss_(y_true, y_pred):
        weight = get_weight_matrix_with_reduced_edges(y_true, max_kernel_overlay)
        # avoiding overflow
        epsilon = K.backend.epsilon()
        y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
        logit_y_pred = K.backend.log(y_pred / (1. - y_pred))
        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
               (K.backend.log(1. + K.backend.exp(-K.backend.abs(logit_y_pred))) + K.backend.maximum(-logit_y_pred, 0.))
        return K.backend.sum(loss) / K.backend.sum(weight)
    return adjusted_weighted_bce_loss_

# weight: weighted tensor(same shape with mask image)
def weighted_bce_loss(y_true, y_pred):
    weight = get_weight_matrix(y_true)
    # avoiding overflow
    epsilon = K.backend.epsilon()
    y_pred = K.backend.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.backend.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
           (K.backend.log(1. + K.backend.exp(-K.backend.abs(logit_y_pred))) + K.backend.maximum(-logit_y_pred, 0.))
    return K.backend.sum(loss) / K.backend.sum(weight)


def weighted_dice_loss(y_true, y_pred):
    weight = get_weight_matrix(y_true)
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 0.0 * weighted_bce_loss(y_true, y_pred, weight) + \
           weighted_dice_loss(y_true, y_pred, weight)
    return loss


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


def surficenDiceLoss(y_true, y_pred):
    alpha_ = alpha
    dice = dice_loss(y_true, y_pred)
    dice *= alpha_
    surface = surface_loss(y_true, y_pred)
    surface *= (1.0 - alpha_)
    return dice + surface


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer  # might be 1 -


def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)  # might be 1 -


def FocalLoss(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.0
    alpha = 0.25

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dice_score(y_true, y_pred):
    smooth = K.backend.epsilon()
    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = K.backend.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.backend.sum(y_true_f) + K.backend.sum(y_pred_f) + smooth)
    return answer


def dice_loss(y_true, y_pred):
    answer = 1. - dice_score(y_true, y_pred)
    return answer


# not working
def Active_Contour_Loss(y_true, y_pred):
    # y_pred = K.cast(y_pred, dtype = 'float64')
    """
    lenth term
    """
    x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2] ** 2
    delta_y = y[:, :, :-2, 1:] ** 2
    delta_u = K.abs(delta_x + delta_y)
    epsilon = 0.00000001  # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon))  # equ.(11) in the paper
    """
    region term
    """
    C_1 = np.ones((480, 320))
    C_2 = np.zeros((480, 320))

    region_in = K.abs(K.sum(y_pred[:, 0, :, :] * ((y_true[:, 0, :, :] - C_1) ** 2)))  # equ.(12) in the paper
    region_out = K.abs(K.sum((1 - y_pred[:, 0, :, :]) * ((y_true[:, 0, :, :] - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.

    loss = lenth + lambdaP * (region_in + region_out)

    return loss
