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

def get_best_weight_name(weight_paths, epoch):
    for weight_path in weight_paths:
        weight_name = os.path.basename(weight_path)
        print(weight_name)
        epoch_number_in_name = weight_name[17:20]
        epoch_number_in_name = epoch_number_in_name.lstrip("0")
        print(epoch_number_in_name)
        epoch_number = (int)(epoch_number_in_name)
        if (epoch == epoch_number):
            return weight_path

def copy_files(source, destination):
    print('Copying from: ' + source)
    print('Copying to:' + destination)
    src_files = os.listdir(source)
    for file_name in src_files:
        full_file_name = os.path.join(source, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination)

def copy_file(file, destination):
    print('Copying to:' + destination)
    if os.path.isfile(file):
        shutil.copy(file, destination)

def main():
    sets = ['Set_0', 'Set_1', 'Set_2', 'Set_3', 'Set_4']
    all_lines = []
    for set in sets:
        set_lines = []
        print('\n' + set + '\n')
        architecturesInputDir = 'D:/CracksTrainings/' + set + '/'
        outputDir = 'D:/CrackTrainings_best/' + set + '/'
        # Get subdirectories of all architectures
        inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
        directory_names = []
        full_paths = []
        for inputArchitectureSurDir in inputArchitecturesSubDirs:
            if 'SEQ' in inputArchitectureSurDir:
                continue #skip
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
            print('Best dice score' + str(score))
            #to check
            receivedDice = currentBenchmark.GetDiceAt(epoch)
            print('Received dice score' + str(receivedDice))
            print('acc rec precision iou dice')
            acc = currentBenchmark.GetAccuracyAt(epoch)
            rec = currentBenchmark.GetRecallAt(epoch)
            pre = currentBenchmark.GetPrecisionAt(epoch)
            iou = currentBenchmark.GetIoUAt(epoch)
            dice = currentBenchmark.GetDiceAt(epoch)
            line = str(acc) + ',' + str(rec) + ',' + str(pre) + ',' + str(iou) + ',' + str(dice)
            set_lines.append(line)

            #add 1 cause epoch starts to count from 1
            full_path = inputArchitectureSurDir + 'prediction//' + str(epoch) + '//'
            full_paths.append(full_path)
            # make subdirectory in the output directory
            best_weights_dir = outputDir + dir_name + '/'
            if not os.path.exists(best_weights_dir):
                print('Output directory doesnt exist!\n')
                print('It will be created in ' + best_weights_dir + '!\n')
                os.makedirs(best_weights_dir)
            copy_files(full_path, best_weights_dir)

            #collect weights names as well
            weights_names = glob.glob(inputArchitectureSurDir + '*.hdf5')
            best_weight_path = get_best_weight_name(weights_names, epoch)
            copy_file(best_weight_path, best_weights_dir)
            input_file.close()
        all_lines.append(set_lines)
    print('Done!')
    #print all
    print('acc rec precision iou dice')
    for set_lines in all_lines:
        for best_scores in set_lines:
            print(best_scores)
        print('\n')
        #makeGraph(x, y)


if __name__ == "__main__":
    main()