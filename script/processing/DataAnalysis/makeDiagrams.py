import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmark import Benchmark
from statistics import Statistics
from render import Render
from imageData import ImageData

def MakeVideo(classNr, configuration, predictionPath):
        inputDir = 'C:/Users/rytis/OneDrive/Desktop/Straipsniai/DAGM/Class' + str(classNr) + '/Test/'
        inputImageDir = inputDir + 'image/'
        inputLabelDir = inputDir + 'label/'
        inputPredictionDir = predictionPath

        imageData = ImageData()
        imageData.load(inputImageDir, inputLabelDir, inputPredictionDir)
        
        outputDirectory = 'C:/Users/rytis/OneDrive/Desktop/Straipsniai/Pattern recognition letters/videoRenders/' 
        if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)

        fileName = ""
        recallSum = 0.0
        precisionSum = 0.0
        accuracySum = 0.0
        f1Sum = 0.0
        IoUSum = 0.0
        dicsSum = 0.0

        outputImageWidth = 1536
        outputImageHeight = 384 + 512
        width = 512
        height = 512

        mainTitle = 'Class' + str(classNr) + ' Network-' + configuration

        architectureVideoName = outputDirectory + mainTitle + '.avi'
        architectureOutVideo = cv2.VideoWriter(architectureVideoName,cv2.VideoWriter_fourcc('X','V','I','D'), 30.0, (outputImageWidth,outputImageHeight))

        for i in range(0, imageData.getDataCount()):
                imagePath, labelPath, predictionPath = imageData.getImageData(i)
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
                outerColor = (80,80,200)
                innerColor = (50,50,140)
                renderedPrediction = Render.Defects(image, prediction, outerColor, innerColor)
                
                #render label on original image
                outerColor = (0,0,255)
                innerColor = (0,0,0)
                renderedImage = Render.Defects(image, label, outerColor, innerColor)

                #cast greyscale to rgb
                prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)

                #form output
                outputImage = np.zeros((outputImageHeight, outputImageWidth,3), np.uint8)

                #image offset from top
                imageOffsetFromTop = 256
                textOffset = 15

                imagebottom = imageOffsetFromTop + height
                outputImage[imageOffsetFromTop:imagebottom, 0:width] = renderedImage
                outputImage[imageOffsetFromTop:imagebottom, width:width*2] = prediction
                outputImage[imageOffsetFromTop:imagebottom, width*2:width*3] = renderedPrediction
                #draw rectange to indentify images boarders
                color = (128, 10, 170)
                cv2.rectangle(outputImage, (0, imageOffsetFromTop), (width, height + imageOffsetFromTop), color, 2)
                cv2.rectangle(outputImage, (width, imageOffsetFromTop), (width * 2, height + imageOffsetFromTop), color, 2)
                cv2.rectangle(outputImage, (width * 2, imageOffsetFromTop), (width * 3, height + imageOffsetFromTop), color, 2)

                #render names for images
                resultsTextThickness = 1
                resultsTextScale = 1.1
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontColor  = (255,255,255)
                resultsTextThickness = 2
                lineType  = cv2.LINE_AA
                #render results under
                currentResultsTitle = 'Image + Label'
                textPos = (120, imageOffsetFromTop - textOffset)
                cv2.putText(outputImage, currentResultsTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                currentResultsTitle = 'Prediction'
                textPos = (682, imageOffsetFromTop - textOffset)
                cv2.putText(outputImage, currentResultsTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                currentResultsTitle = 'Image + Prediction'
                textPos = (1110, imageOffsetFromTop - textOffset)
                cv2.putText(outputImage, currentResultsTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                #render main title
                resultsTextScale = 1.8
                mainTitle = 'Class' + str(classNr) + ' Network-' + configuration
                textPos = (465, 150 - textOffset)
                cv2.putText(outputImage, mainTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                #make it last for 0.5 second
                for i in range(0, 15):
                        architectureOutVideo.write(outputImage)
        
                """
                cv2.imshow('Rendered', renderedPrediction)
                cv2.imshow('Label', label)
                cv2.imshow('Image', renderedImage)
                cv2.imshow('Prediction', prediction)
                """
                cv2.imshow('Output', outputImage)
                cv2.waitKey(1)
        architectureOutVideo.release()

def GetCurrentPredictionInfo(path):
    info = os.path.basename(os.path.normpath(path))
    #three first symbols discribes epoch, then '-' and the rest 6 symbols are dice coeficient
    words = info.split('-')
    epochNumber = int(words[0])
    diceLoss = float(words[1]) 
    return epochNumber, diceLoss

def render(benchmark):
        #get best dice coeficient epoch index
        dice, index = benchmark.GetBestDice()
        configName = benchmark.configName
        classNumber = benchmark.classNr
        neuralNetworkPredictionPath = "C:/Users/rytis/OneDrive/Desktop/Straipsniai/Pattern recognition letters/biggerBatchSizeTraining_output/biggerBatchSizeTraining_output/"
        pathToClassConfigOutputs = neuralNetworkPredictionPath + str(configName) + '/class' + str(classNumber) + '/'
        allEpochsDataPaths = glob.glob(pathToClassConfigOutputs + '*/')
        for path in allEpochsDataPaths:
                dirname = os.path.basename(os.path.dirname(path))
                epochNumber, diceLoss = GetCurrentPredictionInfo(dirname)
                epochNumber = epochNumber - 1
                if epochNumber == index:
                        #open and render
                        MakeVideo(classNumber, configName, path)

def main():
        minAreaFilters = [0, 1]
        configurationNames = ['l2k8', 'l2k16', 'l2k32', 'l3k8', 'l3k16', 'l3k32', 'l4k8','l4k16', 'l4k32', 'l5k8', 'l5k16', 'l5k32']
        parametersCount = [25929, 102025, 117673, 467785, 483433, 1928393, 1944041, 7765961]
        epochTrainingTime = [32, 35, 55, 36, 45, 74, 39, 51, 88, 42, 56, 102]
        iterationExecution = [0.01577, 0.0171, 0.0221, 0.01804, 0.0203, 0.0286, 0.01869, 0.022, 0.0328, 0.02083, 0.0231, 0.037]
        iterationExecutionMS = [15.8, 17.1, 22.1, 18.0, 20.3, 28.6, 18.7, 22.0, 32.8, 20.8, 23.1, 37.0]
        classNames = ['1', '2', '3', '4', '5', '6']

        #####################################################
        #configuration read
        #####################################################

        architecturesInputDir = 'C:/Users/rytis/OneDrive/Desktop/Straipsniai/Pattern recognition letters/retrain/'

        #Get subdirectories of all architectures
        inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')

        results = []

        #make double array
        for i in range(0,6):
                classBenchmark = []
                results.append(classBenchmark)

        configCount = 0
        for inputArchitectureDir in inputArchitecturesSubDirs:
                configName = configurationNames[configCount]
                #do analysis for every class
                for classNr in range(1, 7):
                        #make results for predictions
                        #################################################
                        inputPredictionDir = inputArchitectureDir + str(classNr) + '/'

                        #Get subdirectories from prediction images
                        inputPredictionSubDirs = glob.glob(inputPredictionDir + '*/')

                        txtPath = inputPredictionDir + 'averageScore' + '_minAreastr_0' + '.txt'
                        inputFile = open(txtPath,'r')
                        currentBenchmark = Benchmark()
                        currentBenchmark.classNr = classNr
                        currentBenchmark.configName = configName
                        currentBenchmark.minAreaFilter = 0
                        count = 0
                        for lineText in inputFile:
                                #skip first line
                                if count == 0:
                                        count = count + 1
                                        continue
                                currentBenchmark.parseDataLine(lineText)
                        render(currentBenchmark)
                        results[classNr - 1].append(currentBenchmark)
                        inputFile.close()
                configCount = configCount + 1

if __name__== "__main__":
        main()

