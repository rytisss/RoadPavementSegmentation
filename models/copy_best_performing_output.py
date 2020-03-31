import os
import glob
import cv2
import numpy as np
from benchmark.statistics import Statistics
from benchmark.render import Render
from distutils.dir_util import copy_tree


def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def GetCurrentPredictionInfo(path):
    info = os.path.basename(os.path.normpath(path))
    # three first symbols discribes epoch, then '-' and the rest 6 symbols are dice coeficient
    words = info.split('-')
    epochNumber = int(words[1])
    diceLoss = float(words[2])
    return epochNumber, diceLoss


def GetFileName(path):
    fileNameWithExt = path.rsplit('\\', 1)[1]
    fileName, fileExtension = os.path.splitext(fileNameWithExt)
    return fileName


# method to sort directories by number
def SortDirectoriesByNumber(directories):
    last_directory_parts = []
    for i in range(0, len(directories)):
        last_directory_part = os.path.basename(os.path.normpath(directories[i]))
        last_directory_parts.append((int)(last_directory_part))
    last_directory_parts.sort()
    sorted_list = []
    for i in range(0, len(last_directory_parts)):
        for j in range(0, len(directories)):
            last_directory_part = os.path.basename(os.path.normpath(directories[j]))
            number = (int)(last_directory_part)
            if last_directory_parts[i] == number:
                sorted_list.append(directories[j])
                break
    return sorted_list


def AnalyzeArchitecture(prediction_path='', test_data_path='', config = '', dataset = '', output_dir = ''):
    # do analysis for every class
    trainings = prediction_path
    # Get subdirectories from prediction images
    inputPredictionSubDirs = glob.glob(trainings + '*/')
    configName = os.path.basename(os.path.normpath(trainings))
    counter = 0

    highest_dice = 0.0
    highest_dice_dir = ''

    # Do work in every subdirectory
    for inputPredictionSubDir in inputPredictionSubDirs:
        # check if directories exist
        if not os.path.exists(inputPredictionSubDir):
            print('Input prediction directory doesnt exist!\n')
            exit(0)

        imagePath = test_data_path + 'Images/'
        labelsPath = test_data_path + 'Labels/'

        images = gather_image_from_dir(imagePath)
        labels = gather_image_from_dir(labelsPath)
        predictions = gather_image_from_dir(inputPredictionSubDir)

        recallSum = 0.0
        precisionSum = 0.0
        accuracySum = 0.0
        f1Sum = 0.0
        IoUSum = 0.0
        dicsSum = 0.0
        dataCount = len(images)
        for i in range(0, len(images)):
            imagePath = images[i]
            labelPath = labels[i]
            predictionPath = predictions[i]
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            prediction = cv2.imread(predictionPath, cv2.IMREAD_GRAYSCALE)
            _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

            # do analysis
            tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
            recall = Statistics.GetRecall(tp, fn)
            precision = Statistics.GetPrecision(tp, fp)
            accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
            f1 = Statistics.GetF1Score(recall, precision)
            IoU = Statistics.GetIoU(label, prediction)
            dice = Statistics.GetDiceCoef(label, prediction)
            # print('Recall: ' + str(recall) + ', Precision: ' + str(precision) + ', accuracy: ' + str(accuracy) + ', f1: ' + str(f1) + ', IoU: ' + str(IoU) + ', Dice: ' + str(dice))

            recallSum = recallSum + recall
            precisionSum = precisionSum + precision
            accuracySum = accuracySum + accuracy
            f1Sum = f1Sum + f1
            IoUSum = IoUSum + IoU
            dicsSum = dicsSum + dice

            # render defect on image
            outerColor = (0, 0, 200)
            innerColor = (0, 0, 60)
            showImages = False
            if showImages:
                renderedImage = Render.Defects(image, prediction, outerColor, innerColor)

                cv2.imshow('Rendered', renderedImage)
                cv2.imshow('Label', label)
                cv2.imshow('Image', image)
                cv2.imshow('Prediction', prediction)

                cv2.waitKey(1)

        overallRecall = round(recallSum / float(dataCount), 4)
        overallPrecision = round(precisionSum / float(dataCount), 4)
        overallAccuracy = round(accuracySum / float(dataCount), 4)
        overallF1 = round(f1Sum / float(dataCount), 4)
        overallIoU = round(IoUSum / float(dataCount), 4)
        overallDice = round(dicsSum / float(dataCount), 4)
        print('Overall Score:')
        print('Epoch: ' + str(counter) +
              # 'Loss' + str(loss) +
              'Recall: ' + str(overallRecall) +
              ', Precision: ' + str(overallPrecision) +
              ', accuracy: ' + str(overallAccuracy) +
              ', f1: ' + str(overallF1) +
              ', IoU: ' + str(overallIoU) +
              ', Dice: ' + str(overallDice))

        if highest_dice < overallDice:
            highest_dice = overallDice
            highest_dice_dir = inputPredictionSubDir

        # for line
        averageScoreLine = str(counter) + ' ' + str(overallRecall) + ' ' + str(overallPrecision) + ' ' + str(
            overallAccuracy) + ' ' + str(overallF1) + ' ' + str(overallIoU) + ' ' + str(overallDice) + '\n'
        counter += 1
    print('Highest score in ' + config + ' in dataset ' + dataset + ': ' + highest_dice_dir)
    highest_dice_last_dir = os.path.basename(os.path.normpath(highest_dice_dir))
    output_config_dir = output_dir + config + '/' + dataset + '/' + highest_dice_last_dir + '/'
    print('Output directory: ' + output_config_dir)
    if not os.path.exists(output_config_dir):
        print('Output directory doesnt exist!\n')
        print('It will be created in ' + output_config_dir + '\n')
        os.makedirs(output_config_dir)
    copy_tree(highest_dice_dir, output_config_dir)
    g = 7


def main():
    base_dir = 'E:/pavement inspection/'
    dataset_base_dir = 'E:/pavement inspection/datasets/'
    output_dir = 'C:/Users/Rytis/Desktop/pavement_defect_results/'
    folders = ['pretrained_UNet4',
               'pretrained_UNet4_res',
               'pretrained_UNet4_res_aspp',
               'pretrained_UNet4_res_aspp_AG',
               'pretrained_Unet4_res_asppWF',
               'pretrained_Unet4_res_asppWF_AG']

    folders = ['pretrained_Unet4_res_asppWF_AG']

    directories_to_investigate = []
    for folder in folders:
        config = folder
        #gather all subdirs
        path = base_dir + folder + '/'
        dataset_result_dirs = glob.glob(path + '*/')
        for dataset_result_dir in dataset_result_dirs:
            dataset_result_basename = os.path.basename(os.path.normpath(dataset_result_dir))
            prediction_dir = dataset_result_dir + 'output/'
            if 'crackForest' in dataset_result_basename:
                dataset_dir = dataset_base_dir + 'CrackForestdatasets_output/Test/'
                set_name = 'crackForest'
                AnalyzeArchitecture(prediction_dir, dataset_dir, config, set_name, output_dir)
            elif 'crack500' in dataset_result_basename:
                dataset_dir = dataset_base_dir + 'crack500_out_0.25percent_size/Test/'
                set_name = 'crack500'
                AnalyzeArchitecture(prediction_dir, dataset_dir, config, set_name, output_dir)
            elif 'gaps384' in dataset_result_basename:
                dataset_dir = dataset_base_dir + 'GAPs384_output_0.5percent_size/Test/'
                set_name = 'gaps384'
                AnalyzeArchitecture(prediction_dir, dataset_dir, config, set_name, output_dir)
            else:
                continue


if __name__ == "__main__":
    main()