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
    epochNumber = int(words[0])
    diceLoss = float(words[1]) 
    return epochNumber, diceLoss

def GetFileName(path):
    fileNameWithExt = path.rsplit('\\', 1)[1]
    fileName, fileExtension = os.path.splitext(fileNameWithExt)
    return fileName

averageDefectArea = [9292, 2919, 3455, 7026, 4709, 21038]
minAreaFilters = [0, 1, 5]

architecturesInputDir = 'C:/Users/Rytis/Desktop/biggerBatchSizeTraining_output/'

#Get subdirectories of all architectures
inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')

epochNumberList = [[1, 6, 54, 9, 33, 50, 10, 18, 96, 23, 6, 6], [48, 6, 99, 50, 41, 50, 17, 69, 89, 85, 84, 89], [16, 64, 47, 27, 62, 56, 62, 89, 89, 92, 75, 86], [83, 95, 87, 21, 16, 82, 38, 16, 79, 99, 86, 27], [64, 79, 82, 91, 46, 21, 90, 91, 61, 83, 95, 75], [37, 11, 67, 15, 61, 24, 10, 30, 61, 20, 26, 20]]
#epochNumber1FList = [[17, 17, 33, 97, 84, 93, 80, 60], [72, 86, 99, 44, 61, 44, 97, 82], [90, 53, 43, 59, 62, 86, 79, 96], [82, 83, 74, 69, 94, 69, 94, 89], [96, 56, 31, 24, 43, 65, 89, 95], [87, 35, 88, 95, 90, 95, 45, 91]]
configurationNames = ['l2k8', 'l2k16', 'l2k32', 'l3k8', 'l3k16', 'l3k32', 'l4k8', 'l4k16', 'l4k32', 'l5k8', 'l5k16', 'l5k32']

for classNr in range(1, 7):
    #do analysis for every class
    counter = 0
    for inputArchitectureDir in inputArchitecturesSubDirs:
        #make results for predictions
        #################################################
        #inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class1/Test/'
        inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Test/'
        inputImageDir = inputDir + 'image/'
        inputLabelDir = inputDir + 'label/'
        inputPredictionDir = inputArchitectureDir + 'class' + str(classNr) + '/'

        #Get subdirectories from prediction images
        inputPredictionSubDirs = glob.glob(inputPredictionDir + '*/')

        minAreaFilter = 0.0
        #Do work in every subdirectory
        for inputPredictionSubDir in inputPredictionSubDirs:
            epochNumber, trainingDiceLoss = GetCurrentPredictionInfo(inputPredictionSubDir)
            if epochNumber == epochNumberList[classNr - 1][counter]:
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

                outputDirectory = 'C:/src/Renders/'
                #create directory for image save
                imageOutputDirectory = outputDirectory + str(classNr) + '/' + configurationNames[counter] + '/' + str(minAreaFilter) + ' withoutFilter' '/' 
                if not os.path.exists(imageOutputDirectory):
                    os.makedirs(imageOutputDirectory)
                    
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
                    outerColor = (80,80,200)
                    innerColor = (50,50,140)
                    renderedImage = Render.Defects(image, prediction, outerColor, innerColor)
                    """ 
                    cv2.imshow('Rendered', renderedImage)
                    cv2.imshow('Label', label)
                    cv2.imshow('Image', image)
                    cv2.imshow('Prediction', prediction)
                    cv2.waitKey(1)
                    """
                    #save rendered image
                    cv2.imwrite(imageOutputDirectory + GetFileName(predictionPath) + '.bmp', renderedImage)

                
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

                print('Done!')
        counter = counter + 1
