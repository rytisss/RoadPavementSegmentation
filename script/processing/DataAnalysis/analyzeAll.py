import os
import glob
import cv2
import numpy as np
from imageData import ImageData
from statistics import Statistics
from render import Render
from filtering import Filtering

from joblib import Parallel, delayed

def GetCurrentPredictionInfo(path):
    info = os.path.basename(os.path.normpath(path))
    #three first symbols discribes epoch, then '-' and the rest 6 symbols are dice coeficient
    words = info.split('-')
    epochNumber = int(words[0])
    diceLoss = float(words[1]) 
    return epochNumber, diceLoss

def GetFileName(path):
    fileNameWithExt = path.rsplit('\\', 1)[1]
    fileName, fileExtension = os.path.splitext(fileNameWithExt)
    return fileName

def AnalyzeArchitecture(architectureDir):
    if True:
        #do analysis for every class
        for classNr in range(1, 7):
            #make results for predictions
            #################################################
            #inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class1/Test/'
            inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Test/'
            inputImageDir = inputDir + 'image/'
            inputLabelDir = inputDir + 'label/'
            architectureName = os.path.basename(os.path.normpath(architectureDir))
            txtOutDir = 'C:/Users/Rytis/Desktop/retrain/' + architectureName + '/' + str(classNr) + '/'
            inputPredictionDir = architectureDir + 'class' + str(classNr) + '/'
             #check if directories exist
            if not os.path.exists(txtOutDir):
                print('Output directory doesnt exist!\n')
                print('It will be created!\n')
                os.makedirs(txtOutDir)
            averageDefectArea = [9292, 2919, 3455, 7026, 4709, 21038]
            minAreaFilters = [0, 1]

            #Get subdirectories from prediction images
            inputPredictionSubDirs = glob.glob(inputPredictionDir + '*/')

            for minAreaFilter in minAreaFilters:
                #file for average score of every epoch
                averageScore = open(txtOutDir + 'averageScore' + '_minAreastr_' + str(minAreaFilter) + '.txt','w')
                #first line will be labels of data
                firstLine = 'Epoch TrainingDice Recall Precision Accuracy F1 IoU Dice' + '\n'
                averageScore.write(firstLine)

                #Do work in every subdirectory
                for inputPredictionSubDir in inputPredictionSubDirs:
                    epochNumber, trainingDiceLoss = GetCurrentPredictionInfo(inputPredictionSubDir)
                    trainingDice = round(1. - trainingDiceLoss, 4)
                    print('Epoch: ' + str(epochNumber) + ', training set Dice Coef: ' + str(trainingDice))

                    #check if directories exist
                    if not os.path.exists(inputPredictionSubDir):
                        print('Input prediction directory doesnt exist!\n')
                        exit(0)

                    if not os.path.exists(inputImageDir):
                        print('Input image directory doesnt exist!\n')
                        exit(0)

                    if not os.path.exists(inputLabelDir):
                        print('Input label directory doesnt exist!\n')
                        exit(0)

                    """
                    #create directory for image save
                    imageOutputDirectory = inputPredictionSubDir + 'renderImage_' + str(minAreaFilter) + '/'
                    if not os.path.exists(imageOutputDirectory):
                        os.makedirs(imageOutputDirectory)
                    """
                    
                    imageProvider = ImageData()
                    imageProvider.load(inputImageDir, inputLabelDir, inputPredictionSubDir)
                    print('Data loaded!')
                    dataCount = imageProvider.getDataCount()
                    print(str(dataCount) + ' found!')
                    recallSum = 0.0
                    precisionSum = 0.0
                    accuracySum = 0.0
                    f1Sum = 0.0
                    IoUSum = 0.0
                    dicsSum = 0.0
                    for i in range(dataCount):
                        imagePath, labelPath, predictionPath = imageProvider.getImageData(i)
                        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                        label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)      
                        _, label = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
                        prediction = cv2.imread(predictionPath, cv2.IMREAD_GRAYSCALE)
                        _, prediction = cv2.threshold(prediction,127,255,cv2.THRESH_BINARY)
                        #apply minimum area filter
                        minArea = averageDefectArea[classNr - 1] * minAreaFilter / 100.0
                        prediction = Filtering.MinArea(prediction, minArea)

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
                        outerColor = (0,0,255)
                        innerColor = (0,0,60)
                        #renderedImage = Render.Defects(image, prediction, outerColor, innerColor)
                        
                        """
                        cv2.imshow('Rendered', renderedImage)
                        cv2.imshow('Label', label)
                        cv2.imshow('Image', image)
                        cv2.imshow('Prediction', prediction)

                        cv2.waitKey(1)
                        """
                        #save rendered image
                        #cv2.imwrite(imageOutputDirectory + GetFileName(predictionPath) + '.jpg', renderedImage)

                    overallRecall = round(recallSum / float(dataCount), 4)
                    overallPrecision = round(precisionSum / float(dataCount), 4)
                    overallAccuracy = round(accuracySum / float(dataCount), 4)
                    overallF1 = round(f1Sum / float(dataCount), 4)
                    overallIoU = round(IoUSum / float(dataCount), 4)
                    overallDice = round(dicsSum / float(dataCount), 4)
                    print('Overall Score:')
                    print('Recall: ' + str(overallRecall) + 
                    ', Precision: ' + str(overallPrecision) +
                    ', accuracy: ' + str(overallAccuracy) + 
                    ', f1: ' + str(overallF1) + 
                    ', IoU: ' + str(overallIoU) + 
                    ', Dice: ' + str(overallDice))

                    #for line
                    averageScoreLine = str(epochNumber) + ' ' + str(trainingDice) + ' ' + str(overallRecall) + ' ' + str(overallPrecision) + ' ' + str(overallAccuracy) + ' ' + str(overallF1) + ' ' + str(overallIoU) + ' ' + str(overallDice) + '\n'
                    averageScore.write(averageScoreLine)
                    print(inputPredictionDir + ' epoch ' + str(epochNumber) + ' with filter '+ str(minAreaFilter) + ' done!')

                averageScore.close()


def main():
    #architecturesInputDir = 'D:/DagmUNetImageOut/'
    architecturesInputDir = 'C:/Users/Rytis/Desktop/biggerBatchSizeTraining_output/'
    #Get subdirectories of all architectures
    inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
    #for inputDir in inputArchitecturesSubDirs:
    #    AnalyzeArchitecture(inputDir)
    Parallel(n_jobs=8)(delayed(AnalyzeArchitecture)(inputDir) for inputDir in inputArchitecturesSubDirs)

if __name__ == "__main__":
    main()





