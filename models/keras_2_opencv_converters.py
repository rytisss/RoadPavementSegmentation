import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from models.autoencoder import *

def main():
    # Weights path
    weight_path = r'C:\Users\Rytis\Desktop\drill holes inspection\weights\weights\UNet5_16_res_assp_First5x5\UNet5_res_assp_First5x5/DrillSegmentation_UNet5_res_assp_First5x5_72000.hdf5'
     # Choose your 'super-model'
    model = UNet5_res_aspp_First5x5(pretrained_weights=weight_path, number_of_kernels=16, input_size=(480, 480, 1),
                                  loss_function=Loss.CROSSENTROPY)
    # Save model to SavedModel format
    tf.saved_model.save(model, "models/")
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir="./frozen_models",
                          name="frozen_graph.pb",
                          as_text=False)
    print('model saved')
if __name__ == '__main__':
    main()