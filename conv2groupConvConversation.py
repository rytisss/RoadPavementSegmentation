import tensorflow as tf
import numpy as np

def transferConvToGroupConv(model, model_groupConv):
    latest_group_conv2D_layer_index = 0
    for first_model_index in range(0, len(model.layers)):
        # check types: first should be conv2d, second - deformable conv2d
        name_1 = model.layers[first_model_index].__class__.__name__
        if name_1 == 'Conv2D' or name_1 == 'BatchNormalization' or name_1 == 'Conv2D':
            for second_model_index in range(latest_group_conv2D_layer_index, len(model_groupConv.layers)):
                name_2 = model_groupConv.layers[second_model_index].__class__.__name__
                if name_1 == 'Conv2D':
                    if name_2 == 'Conv2D':
                        first_weights = model.layers[first_model_index].get_weights()
                        second_weights = model_groupConv.layers[second_model_index].get_weights()
                        model_groupConv.layers[second_model_index].set_weights(first_weights)
                        latest_group_conv2D_layer_index = second_model_index
                        break
                    elif name_2 == 'GroupConv2D':
                        first_weights = model.layers[first_model_index].get_weights()
                        second_weights = model_groupConv.layers[second_model_index].get_weights()
                        if first_weights[0].shape[2] == second_weights[0].shape[2]:
                            model_groupConv.layers[second_model_index].set_weights(first_weights)
                        else:
                            for i in range(0, first_weights[0].shape[3]):
                                for j in range(0, first_weights[0].shape[2]):
                                    feature_kernel = first_weights[0][:,:,j,i]
                                    number_of_rotations = 4
                                    for rotation_index in range(0, number_of_rotations):
                                        index_in_second_weights = rotation_index * first_weights[0].shape[2] + rotation_index
                                        second_weights[0][:,:,index_in_second_weights,i] = np.rot90(feature_kernel,k=rotation_index)
                                        #print(index_in_second_weights)
                                        #print(feature_kernel)
                                        #print(rotation_index * 90)
                                        #print(second_weights[0][:,:,index_in_second_weights,i])
                                        #print(50 * '-')
                            model_groupConv.layers[second_model_index].set_weights(second_weights)
                        latest_group_conv2D_layer_index = second_model_index
                        break
                    else:
                        continue
                if name_1 == 'BatchNormalization':
                    if name_2 == 'BatchNormalization':
                        first_weights = model.layers[first_model_index].get_weights()
                        second_weights = model_groupConv.layers[second_model_index].get_weights()
                        model_groupConv.layers[second_model_index].set_weights(first_weights)
                        latest_group_conv2D_layer_index = second_model_index
                        break
                    else:
                        continue

def transferConvToDeformedConv(model, model_deformedConv):
    latest_group_conv2D_layer_index = 0
    for first_model_index in range(0, len(model.layers)):
        # check types: first should be conv2d, second - deformable conv2d
        name_1 = model.layers[first_model_index].__class__.__name__
        if name_1 == 'Conv2D' or name_1 == 'BatchNormalization' or name_1 == 'Conv2D':
            for second_model_index in range(latest_group_conv2D_layer_index, len(model_deformedConv.layers)):
                name_2 = model_deformedConv.layers[second_model_index].__class__.__name__
                if name_1 == 'Conv2D':
                    if name_2 == 'Conv2D':
                        first_weights = model.layers[first_model_index].get_weights()
                        second_weights = model_deformedConv.layers[second_model_index].get_weights()
                        model_deformedConv.layers[second_model_index].set_weights(first_weights)
                        latest_group_conv2D_layer_index = second_model_index
                        break
                    elif name_2 == 'DeformableConv2D':
                        first_weights = model.layers[first_model_index].get_weights()
                        second_weights = model_deformedConv.layers[second_model_index].get_weights()
                        for i in range(0, first_weights[0].shape[2]):
                            for j in range(0, first_weights[0].shape[3]):
                                index_in_second = i * first_weights[0].shape[3] + j
                                second_weights[0][:,:,index_in_second,0] = first_weights[0][:,:,i,j]
                                model_deformedConv.layers[second_model_index].set_weights(second_weights)
                        latest_group_conv2D_layer_index = second_model_index
                        break
                    else:
                        continue
                if name_1 == 'BatchNormalization':
                    if name_2 == 'BatchNormalization':
                        first_weights = model.layers[first_model_index].get_weights()
                        second_weights = model_deformedConv.layers[second_model_index].get_weights()
                        model_deformedConv.layers[second_model_index].set_weights(first_weights)
                        latest_group_conv2D_layer_index = second_model_index
                        break
                    else:
                        continue




