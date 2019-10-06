import os
import glob
import cv2
import numpy as np
from augmentationTool import AugmentationTool

for classNr in range(1,7):
    #################################################
    inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Train/'
    inputLabelDir = inputDir + 'Label/'
    outputDir = inputDir + 'Augmented/'

    #check if directories exist
    if not os.path.exists(inputDir):
        print('Input directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(inputLabelDir):
        print('Input label directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(outputDir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputDir)

    #Collect *png files in input directory
    inputImageList = glob.glob(inputDir + '*.PNG')
    inputImageLabelList = glob.glob(inputLabelDir + '*.PNG')

    #Augmentation
    rotationAngleList = [0, 90, 180, 270]
    for imagePath in inputImageList:
        imageNameWithExt = imagePath.rsplit('\\', 1)[1]
        imageName, imageExtension = os.path.splitext(imageNameWithExt)
        #search of image equvalent in 'Labels'
        for imageLabelPath in inputImageLabelList:
            imageLabelNameWithExt = imageLabelPath.rsplit('\\', 1)[1]
            imageLabelName, imageLabelExtension = os.path.splitext(imageLabelNameWithExt)
            #strip '_label' from label image name
            imageLabelNameStripped = imageLabelName.replace('_label', '')
            #now continue until we will find right label to image
            if imageName == imageLabelNameStripped:
                #check label image if it containes defect mark
                print('Opening: ' + imageLabelPath)
                imageLabel = cv2.imread(imageLabelPath, cv2.IMREAD_GRAYSCALE)
                _, imageLabel = cv2.threshold(imageLabel, 230, 255, cv2.THRESH_BINARY)
                cv2.imshow('tett', imageLabel)
                print('Opening: ' + imagePath)
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if cv2.countNonZero(imageLabel) == 0:
                    #do not augment, just copy
                    #save original image
                    savingPath = outputDir + imageName + '.PNG'
                    print('Saving: ' + savingPath)
                    cv2.imwrite(savingPath, image)
                    #save label, just change '_label' to '_mask_
                    savingPath = outputDir + imageLabelNameStripped + '_label' + '.PNG'
                    print('Saving: ' + savingPath)
                    cv2.imwrite(savingPath, imageLabel)
                    break
                else:
                    #augment
                    for angle in rotationAngleList:
                        print('Rotating in ' + str(angle))
                        rotatedImage = AugmentationTool.RotateImage(image, angle)
                        savingPath = outputDir + imageName + '_rot' + str(angle) + '.PNG'
                        print('Saving: ' + savingPath)
                        cv2.imwrite(savingPath, rotatedImage)
                        cv2.imshow('Original Image', rotatedImage)

                        print('Rotating in ' + str(angle))
                        rotatedLabelImage = AugmentationTool.RotateImage(imageLabel, angle)
                        savingPath = outputDir + imageLabelNameStripped + '_rot' + str(angle) + '_label' + '.PNG'
                        print('Saving: ' + savingPath)
                        cv2.imwrite(savingPath, rotatedLabelImage)
                        cv2.imshow('Augmented Image', rotatedLabelImage)

                        cv2.waitKey(1)
            
                    #now flip horizontally and do same rotation
                    flipImage = AugmentationTool.FlipImageHorizontally(image)
                    flipImageLabel = AugmentationTool.FlipImageHorizontally(imageLabel)

                    for angle in rotationAngleList:
                        print('Rotating in ' + str(angle))
                        rotatedImage = AugmentationTool.RotateImage(flipImage, angle)
                        savingPath = outputDir + imageName + '_rotFlip' + str(angle) + '.PNG'
                        print('Saving: ' + savingPath)
                        cv2.imwrite(savingPath, rotatedImage)
                        cv2.imshow('Original Image', rotatedImage)

                        print('Rotating in ' + str(angle))
                        rotatedLabelImage = AugmentationTool.RotateImage(flipImageLabel, angle)
                        savingPath = outputDir + imageLabelNameStripped + '_rotFlip' + str(angle) + '_label' +'.PNG'
                        print('Saving: ' + savingPath)
                        cv2.imwrite(savingPath, rotatedLabelImage)
                        cv2.imshow('Augmented Image', rotatedLabelImage)

                        cv2.waitKey(1)

                    break

    print('Done!')