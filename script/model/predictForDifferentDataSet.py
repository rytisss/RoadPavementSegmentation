import os
import glob
import shutil
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from script.model.utilities import *
from script.model.autoencoder import *
import keras as K

def predict_and_save(weight_path, output_path):
    size = (320, 480, 1)
    model = AutoEncoder4_5x5(number_of_kernels=32, input_size=size, pretrained_weights=weight_path,
                             loss_function=Loss.CROSSENTROPY)
    testGene = testGenerator('D:/Sensors_FromIDAACS/major revision/another dataset comparisson/Images/',
                             target_size=(320, 480))
    results = model.predict_generator(testGene, 1233, verbose=1)
    predictionOutputDir = output_path
    if not os.path.exists(predictionOutputDir):
        os.makedirs(predictionOutputDir)
    saveResult(predictionOutputDir, results)

def main():
    sets = ['Set_0', 'Set_1', 'Set_2', 'Set_3', 'Set_4']
    for set in sets:
        architecturesInputDir = 'D:/CrackTrainings_best/' + set + '/'
        outputDir = 'D:/CrackTrainings_best/' + set + '/'
        # Get subdirectories of all architectures
        inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
        directory_names = []
        full_paths = []
        for inputArchitectureSurDir in inputArchitecturesSubDirs:
            if 'SEQ' in inputArchitectureSurDir:
                continue #skip
            weight_paths = glob.glob(inputArchitectureSurDir + '*.hdf5')
            output_path = inputArchitectureSurDir + 'crack200_new/'
            print(weight_paths[0])
            #test and save
            predict_and_save(weight_paths[0], output_path)

if __name__ == "__main__":
    main()