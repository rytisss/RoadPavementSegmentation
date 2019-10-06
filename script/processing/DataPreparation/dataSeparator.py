import os
import glob
import cv2
import numpy as np
from augmentationTool import AugmentationTool

#basically takes data and puts it to different folders
#################################################
for classNr in range(1,7):
    inputDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Test/'
    outputImageDir = inputDir + 'image/'
    outputLabelDir = inputDir + 'label/'

    prefix = '_label'
    #saving format
    extension = '.png'

    #check if directories exist
    if not os.path.exists(inputDir):
        print('Input directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(outputImageDir):
        print('Output image directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputImageDir)

    if not os.path.exists(outputLabelDir):
        print('Output label directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputLabelDir)

    #Collect *png files in input directory
    inputImageList = glob.glob(inputDir + '*.PNG')

    for imagePath in inputImageList:
        imageNameWithExt = imagePath.rsplit('\\', 1)[1]
        imageName, imageExtension = os.path.splitext(imageNameWithExt)
        #search of image equvalent in 'Labels'
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        if prefix in imagePath:
            imageName = imageName.replace(prefix, '')
            path = outputLabelDir + imageName + extension
            cv2.imwrite(path, image)
            cv2.imshow('Label', image)
        else:
            path = outputImageDir + imageName + extension
            cv2.imwrite(path, image)
            cv2.imshow('Image', image)
        cv2.waitKey(1)

    print('Done!')