import os
import glob
import cv2
import numpy as np
import random

"""
Searches for an index of label for image
Label should be with same name + 'additionalExtention'
"""
def GetLabelFilePath(imageFileName, labelFilesList, additionalExtention):
    indexCounter = 0
    for labelFile in labelFilesList:
        labelNameWithExt = labelFile.rsplit('\\', 1)[1]
        labelName, labelExtension = os.path.splitext(labelNameWithExt)
        #substract 'additionalExtention' from labelName
        labelNameWithoutExtension = labelName.replace(additionalExtention, '')
        if imageFileName == labelNameWithoutExtension:
            return indexCounter
        indexCounter+=1
    return -1 #nothing is found


#set how many set needed
set2Create = 5
#percentage of set as test
tesSetPercentage = 10
#set directories
inputImageDir = 'C:/Users/DeepLearningRig/Desktop/crackForestDataset/Images/'
inputLabelDir = 'C:/Users/DeepLearningRig/Desktop/crackForestDataset/Labels/'
outputDir = 'E:/RoadCracksInspection/datasets90-10/'

#check if directories exist
if not os.path.exists(inputImageDir):
    print('Image input directory doesnt exist!\n')
    exit(0)
if not os.path.exists(inputLabelDir):
    print('Label input directory doesnt exist!\n')
    exit(0)

inputImageList = glob.glob(inputImageDir + '*.bmp')
inputLabelList = glob.glob(inputLabelDir + '*.bmp')

imageCount = min(len(inputImageList), len(inputLabelList))
trainingSetImageCount = (int)(imageCount * tesSetPercentage / 100)

for setIndex in range(0, set2Create):
    #randomly shuffle each time
    random.shuffle(inputImageList)
    for i in range(0, imageCount):
        #image path
        imagePath = inputImageList[i]
        imageNameWithExt = imagePath.rsplit('\\', 1)[1]
        imageName, extension = os.path.splitext(imageNameWithExt)
        labelIndex = GetLabelFilePath(imageName, inputLabelList, '_label')
        #check if something is found 
        if labelIndex == -1:
            continue #do nothing cause label is not found
        labelPath = inputLabelList[labelIndex]
        labelNameWithExt = labelPath.rsplit('\\', 1)[1]
        #take '_label' from label name
        labelNameWithExt = labelNameWithExt.replace('_label','')
        labelName, extension = os.path.splitext(labelNameWithExt)
    
        print('Opening: ' + imagePath)
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        print('Opening: ' + labelPath)
        label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)
        #show image
        cv2.imshow('image', image)
        cv2.imshow('label', label)
        cv2.waitKey(1)

        setOutputDir = outputDir + "Set_" + str(setIndex) + '//'

        #first will go to testing set, then to training
        if (i < trainingSetImageCount):
            testLabelOutDir = setOutputDir + 'Test//Labels//'
            testImageOutDir = setOutputDir + 'Test//Images//'
            testOutDir = setOutputDir + 'Test//Set//'
            if not os.path.exists(testLabelOutDir):
                os.makedirs(testLabelOutDir)
            if not os.path.exists(testImageOutDir):
                os.makedirs(testImageOutDir)
            if not os.path.exists(testOutDir):
                os.makedirs(testOutDir)
            cv2.imwrite(testLabelOutDir + labelNameWithExt, label)
            cv2.imwrite(testImageOutDir + imageNameWithExt, image)
            cv2.imwrite(testOutDir + labelNameWithExt, label)
            cv2.imwrite(testOutDir + imageNameWithExt, image)
        else:
            trainLabelOutDir = setOutputDir + 'Train//Labels//'
            trainImageOutDir = setOutputDir + 'Train//Images//'
            trainOutDir = setOutputDir + 'Train//Set//'
            if not os.path.exists(trainLabelOutDir):
                os.makedirs(trainLabelOutDir)
            if not os.path.exists(trainImageOutDir):
                os.makedirs(trainImageOutDir)
            if not os.path.exists(trainOutDir):
                os.makedirs(trainOutDir)
            cv2.imwrite(trainLabelOutDir + labelNameWithExt, label)
            cv2.imwrite(trainImageOutDir + imageNameWithExt, image)
            cv2.imwrite(trainOutDir + labelNameWithExt, label)
            cv2.imwrite(trainOutDir + imageNameWithExt, image)


        #save image
        #outputFileName = outputDir + fileName + '.bmp'
        #cv2.imwrite(outputFileName, img)