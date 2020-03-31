import os
import glob
import cv2
import numpy as np
from benchmark.statistics import Statistics
from benchmark.render import Render

def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list

def GetCurrentPredictionInfo(path):
    info = os.path.basename(os.path.normpath(path))
    #three first symbols discribes epoch, then '-' and the rest 6 symbols are dice coeficient
    words = info.split('-')
    epochNumber = int(words[1])
    diceLoss = float(words[2]) 
    return epochNumber, diceLoss

def GetFileName(path):
    fileNameWithExt = path.rsplit('\\', 1)[1]
    fileName, fileExtension = os.path.splitext(fileNameWithExt)
    return fileName

#method to sort directories by number
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

def AnalyzeArchitecture(prediction_path='', test_data_path=''):
    #do analysis for every class
    trainings = prediction_path
    #trainings = 'E:/pavement inspection/lr_scheduler/CrackForest_UNet5_res_aspp/output/'
    allScores = open(trainings + 'allScores.txt','w')

    #Get subdirectories from prediction images
    inputPredictionSubDirs = glob.glob(trainings + '*/')

    #sort directories by number in ascending order
    #inputPredictionSubDirs = SortDirectoriesByNumber(inputPredictionSubDirs)


    configName = os.path.basename(os.path.normpath(trainings))
    firstLine = 'Epoch Recall Precision Accuracy F1 IoU Dice' + '\n'
    allScores.write(configName + '\n')
    allScores.write(firstLine)
        
    #gather all training weights
    #weights = glob.glob(inputDir + '*.hdf5')
    counter = 0
    #Do work in every subdirectory
    for inputPredictionSubDir in inputPredictionSubDirs:
        averageScore = open(inputPredictionSubDir + 'averageScore' + '.txt', 'w')

        averageScore.write(firstLine)
        #from weights name parse epoch index and loss
        #weightPath = weights[counter]
        #weightName = GetFileName(weightPath)
        #epochNr, loss = GetCurrentPredictionInfo(weightName)

        #epochNumber, trainingDiceLoss = GetCurrentPredictionInfo(inputPredictionSubDir)
        #trainingDice = round(1. - trainingDiceLoss, 4)
        #print('Epoch: ' + str(epochNumber) + ', training set Dice Coef: ' + str(trainingDice))

        #file for average score of every epoch
            
        #first line will be labels of data
            
        #check if directories exist
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
            _, label = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
            prediction = cv2.imread(predictionPath, cv2.IMREAD_GRAYSCALE)
            _, prediction = cv2.threshold(prediction,127,255,cv2.THRESH_BINARY)

            #do analysis
            tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
            recall = Statistics.GetRecall(tp, fn)
            precision = Statistics.GetPrecision(tp, fp)
            accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
            f1 = Statistics.GetF1Score(recall, precision)
            IoU = Statistics.GetIoU(label, prediction)
            dice = Statistics.GetDiceCoef(label, prediction)
            #print('Recall: ' + str(recall) + ', Precision: ' + str(precision) + ', accuracy: ' + str(accuracy) + ', f1: ' + str(f1) + ', IoU: ' + str(IoU) + ', Dice: ' + str(dice))

            recallSum = recallSum + recall
            precisionSum = precisionSum + precision
            accuracySum = accuracySum + accuracy
            f1Sum = f1Sum + f1
            IoUSum = IoUSum + IoU
            dicsSum = dicsSum + dice

            #render defect on image
            outerColor = (0,0,200)
            innerColor = (0,0,60)
            renderedImage = Render.Defects(image, prediction, outerColor, innerColor)
                        

            cv2.imshow('Rendered', renderedImage)
            cv2.imshow('Label', label)
            cv2.imshow('Image', image)
            cv2.imshow('Prediction', prediction)

            cv2.waitKey(1)

            #save rendered image
            output_dir = inputPredictionSubDir + 'render/'
            if not os.path.exists(output_dir):
                print('Output directory doesnt exist!\n')
                print('It will be created in ' + output_dir + '\n')
                os.makedirs(output_dir)
            cv2.imwrite(output_dir + GetFileName(predictionPath) + '_render.jpg', renderedImage)

        overallRecall = round(recallSum / float(dataCount), 4)
        overallPrecision = round(precisionSum / float(dataCount), 4)
        overallAccuracy = round(accuracySum / float(dataCount), 4)
        overallF1 = round(f1Sum / float(dataCount), 4)
        overallIoU = round(IoUSum / float(dataCount), 4)
        overallDice = round(dicsSum / float(dataCount), 4)
        print('Overall Score:')
        print('Epoch: ' + str(counter) +
                #'Loss' + str(loss) +
                'Recall: ' + str(overallRecall) +
                ', Precision: ' + str(overallPrecision) +
                ', accuracy: ' + str(overallAccuracy) +
                ', f1: ' + str(overallF1) +
                ', IoU: ' + str(overallIoU) +
                ', Dice: ' + str(overallDice))

                #for line
        averageScoreLine = str(counter) + ' ' + str(overallRecall) + ' ' + str(overallPrecision) + ' ' + str(overallAccuracy) + ' ' + str(overallF1) + ' ' + str(overallIoU) + ' ' + str(overallDice) + '\n'
        averageScore.write(averageScoreLine)
        allScores.write(averageScoreLine)
        counter+=1
        averageScore.close()
    allScores.close()


def main():
    AnalyzeArchitecture()

if __name__ == "__main__":
    main()





