import os
import glob
import cv2
import numpy as np
from imageData import ImageData
from statistics import Statistics
from render import Render
from filtering import Filtering


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

def AnalyzeArchitecture():
    #do analysis for every class
    trainings = 'E:/RoadCracksInspection/trainingOutput/1/'
    allScores = open(trainings + 'allScores.txt','w')

    inputDirs = glob.glob(trainings + '*/')
    for inputDir in inputDirs:
        #Get subdirectories from prediction images
        inputPredictionSubDirs = glob.glob(inputDir + 'prediction/' + '*/')
        averageScore = open(inputDir + 'averageScore' + '.txt','w')
        configName = os.path.basename(os.path.normpath(inputDir))
        firstLine = 'Epoch TrainingLoss Recall Precision Accuracy F1 IoU Dice' + '\n'
        averageScore.write(firstLine)
        allScores.write(configName + '\n')
        allScores.write(firstLine)
        
        #gather all training weights
        weights = glob.glob(inputDir + '*.hdf5')
        counter = 0
        #Do work in every subdirectory
        for inputPredictionSubDir in inputPredictionSubDirs:
            #from weights name parse epoch index and loss
            weightPath = weights[counter]
            weightName = GetFileName(weightPath)
            epochNr, loss = GetCurrentPredictionInfo(weightName)

            #epochNumber, trainingDiceLoss = GetCurrentPredictionInfo(inputPredictionSubDir)
            #trainingDice = round(1. - trainingDiceLoss, 4)
            #print('Epoch: ' + str(epochNumber) + ', training set Dice Coef: ' + str(trainingDice))

            #file for average score of every epoch
            
            #first line will be labels of data
            
            #check if directories exist
            if not os.path.exists(inputPredictionSubDir):
                print('Input prediction directory doesnt exist!\n')
                exit(0)


                """
                #create directory for image save
                imageOutputDirectory = inputPredictionSubDir + 'renderImage_' + str(minAreaFilter) + '/'
                if not os.path.exists(imageOutputDirectory):
                    os.makedirs(imageOutputDirectory)
            
                """
            imagePath = 'E:/RoadCracksInspection/datasets/Set_1/Test/Images/'
            labelsPath = 'E:/RoadCracksInspection/datasets/Set_1/Test/Labels/'
            images = glob.glob(imagePath + '*.bmp')
            labels = glob.glob(labelsPath + '*.bmp')  
            predictions = glob.glob(inputPredictionSubDir + '*.bmp')      
            
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
                innerColor = (0,0,80)
                renderedImage = Render.Defects(image, prediction, outerColor, innerColor)
                        
                
                cv2.imshow('Rendered', renderedImage)
                cv2.imshow('Label', label)
                cv2.imshow('Image', image)
                cv2.imshow('Prediction', prediction)

                cv2.waitKey(1)
                
                        #save rendered image
                        #cv2.imwrite(imageOutputDirectory + GetFileName(predictionPath) + '.jpg', renderedImage)

            overallRecall = round(recallSum / float(dataCount), 4)
            overallPrecision = round(precisionSum / float(dataCount), 4)
            overallAccuracy = round(accuracySum / float(dataCount), 4)
            overallF1 = round(f1Sum / float(dataCount), 4)
            overallIoU = round(IoUSum / float(dataCount), 4)
            overallDice = round(dicsSum / float(dataCount), 4)
            print('Overall Score:')
            print('Epoch: ' + str(epochNr) + 
                    'Loss' + str(loss) + 
                    'Recall: ' + str(overallRecall) + 
                    ', Precision: ' + str(overallPrecision) +
                    ', accuracy: ' + str(overallAccuracy) + 
                    ', f1: ' + str(overallF1) + 
                    ', IoU: ' + str(overallIoU) + 
                    ', Dice: ' + str(overallDice))

                #for line
            averageScoreLine = str(epochNr) + ' ' + str(loss) +  ' ' + str(overallRecall) + ' ' + str(overallPrecision) + ' ' + str(overallAccuracy) + ' ' + str(overallF1) + ' ' + str(overallIoU) + ' ' + str(overallDice) + '\n'
            averageScore.write(averageScoreLine)
            allScores.write(averageScoreLine)
            counter+=1
        averageScore.close()
    allScores.close()


def main():
    AnalyzeArchitecture()

if __name__ == "__main__":
    main()





