import os
import glob
import cv2
import numpy as np
from augmentationTool import AugmentationTool

#################################################
for classNr in range(1,7):
    inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Train/'
    outputDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/TrainAumented/'

    #check if directories exist
    if not os.path.exists(inputDir):
        print('Input directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(outputDir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputDir)

    #Collect *png files in input directory
    inputTrainImageList = glob.glob(inputDir + '*.PNG')

    #Augmentation
    rotationAngleList = [0, 180]
    for imagePath in inputTrainImageList:
        fileNameWithExt = imagePath.rsplit('\\', 1)[1]
        fileName, extension = os.path.splitext(fileNameWithExt)
        print('Opening: ' + imagePath)
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        imgWidth, imgHeight = img.shape
        #check if data contained '_label' index
        #if it does, discard it and add '_mask' in the end of name
        containedPrefix = False
        if '_label' in fileName:
            containedPrefix = True
            fileName = fileName.replace('_label', '')

        if imgWidth > 0 and imgHeight > 0:
            print('Opened succesfully!')
            #rotate in 0, 90, 180, 270 then flip and rotate one more time
            for angle in rotationAngleList:
                print('Rotating in ' + str(angle))
                rotatedImage = AugmentationTool.RotateImage(img, angle)
                savingPath = outputDir + fileName + '_rot' + str(angle)
                if containedPrefix:
                    savingPath = savingPath + '_label'
                savingPath = savingPath  + '.PNG'
                print('Saving: ' + savingPath)
                cv2.imwrite(savingPath, rotatedImage)
                
                #cv2.imshow('Augmented Image', rotatedImage)
                #cv2.waitKey(1)
            
            #now flip horizontally and do same rotation
            flipImage = AugmentationTool.FlipImageHorizontally(img)
            for angle in rotationAngleList:
                print('Rotating flipped image in ' + str(angle))
                rotatedFlippedImage = AugmentationTool.RotateImage(flipImage, angle)
                savingPath = outputDir + fileName + '_flippedRot' + str(angle)
                if containedPrefix:
                    savingPath = savingPath + '_label'
                savingPath = savingPath + '.PNG'
                print('Saving: ' + savingPath)
                cv2.imwrite(savingPath, rotatedFlippedImage)
                
                #cv2.imshow('Augmented Image', rotatedFlippedImage)
                #cv2.waitKey(1)

    print('Done!')



