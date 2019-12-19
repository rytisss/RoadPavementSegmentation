import os
import glob
import shutil
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from script.processing.DataAnalysis.benchmark import Benchmark
from script.processing.DataAnalysis.statistics import Statistics
from script.processing.DataAnalysis.imageData import ImageData

def copy_files(source, destination):
    print('Copying from: ' + source)
    print('Copying to:' + destination)
    src_files = os.listdir(source)
    for file_name in src_files:
        full_file_name = os.path.join(source, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination)

def main():
    sets = ['Set_0', 'Set_1', 'Set_2', 'Set_3', 'Set_4']
    for set in sets:
        architecturesInputDir = 'D:/CracksTrainings/' + set + '/'
        outputDir = 'D:/CrackTrainings_best/' + set + '/'
        # Get subdirectories of all architectures
        inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
        directory_names = []
        dice_scores = []
        epoch_number = []
        full_paths = []
        for inputArchitectureSurDir in inputArchitecturesSubDirs:
            txt_path = inputArchitectureSurDir + 'averageScore.txt'
            dir_name = os.path.basename(os.path.normpath(inputArchitectureSurDir))
            input_file = open(txt_path, 'r')
            currentBenchmark = Benchmark()
            count = 0
            for lineText in input_file:
                # skip first line
                if count == 0:
                    count = count + 1
                    continue
                currentBenchmark.parseDataLine(lineText, count-1)
                count += 1
            directory_names.append(dir_name)
            score, epoch = currentBenchmark.GetBestDice()
            dice_scores.append(score)
            epoch_number.append(epoch)
            full_path = inputArchitectureSurDir + 'prediction//' + str(epoch) + '//'
            full_paths.append(full_path)
            # make subdirectory in the output directory
            best_weights_dir = outputDir + dir_name + '/'
            if not os.path.exists(best_weights_dir):
                print('Output directory doesnt exist!\n')
                print('It will be created in ' + best_weights_dir + '!\n')
                os.makedirs(best_weights_dir)
            copy_files(full_path, best_weights_dir)
            input_file.close()
        print('Done!')
        #makeGraph(x, y)


if __name__ == "__main__":
    main()