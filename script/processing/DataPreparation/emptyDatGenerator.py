import os
import glob
import cv2
import numpy as np

for classNr in range(1,7):
    #################################################
    inputImageDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Train/'
    inputImageLabelsDir = inputImageDir + 'Label/'
    outputImageLabelsDir = inputImageDir + 'Label/'

    #check if directories exist
    if not os.path.exists(inputImageDir):
        print('Input image directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(inputImageLabelsDir):
        print('Input label directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(outputImageLabelsDir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputImageLabelsDir)

    #Collect *png files in input directory
    inputImageList = glob.glob(inputImageDir + '*.PNG')
    inputLabelImageList = glob.glob(inputImageLabelsDir + '*.PNG')

    #Check all name in input list and try to find same name +'_label' in label list
    #If name doesn't exist, create empty black image with same dimension
    for imagePath in inputImageList:
        inputImageNameWithExt = imagePath.rsplit('\\', 1)[1]
        inputImageName, inputImageNameExtension = os.path.splitext(inputImageNameWithExt)
        print('Opening: ' + imagePath)
        inputImageMat = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        imageHeight, imageWidth = inputImageMat.shape
        if inputImageMat.any():
            print('Opened succesfully!')
        else:
            print('Cant open, inspect better...')
            continue
        labelFound = False
        for imageLabelPath in inputLabelImageList:
            inputLabelImageNameWithExt = imageLabelPath.rsplit('\\', 1)[1]
            inputLabelImageName, inputLabelImageNameExtension = os.path.splitext(inputLabelImageNameWithExt)
            #check if image name is in label name
            if inputImageName in imageLabelPath:
                labelFound = True
        #If nothing was found, just make and label image, all black
        if not labelFound:
            blackImage = np.zeros((imageWidth, imageHeight, 1), np.uint8)
            newLabelImageSavingPath = outputImageLabelsDir + inputImageName + '_label.PNG'
            cv2.imwrite(newLabelImageSavingPath, blackImage)
            print('Image saved in ' + newLabelImageSavingPath)
    #THE END
    #################################################

    inputImageDir = 'C:/Users/Rytis/Desktop/DAGM/Class' + str(classNr) + '/Test/'
    inputImageLabelsDir = inputImageDir + 'Label/'
    outputImageLabelsDir = inputImageDir + 'Label/'

    #check if directories exist
    if not os.path.exists(inputImageDir):
        print('Input image directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(inputImageLabelsDir):
        print('Input label directory doesnt exist!\n')
        exit(0)

    if not os.path.exists(outputImageLabelsDir):
        print('Output directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(outputImageLabelsDir)

    #Collect *png files in input directory
    inputImageList = glob.glob(inputImageDir + '*.PNG')
    inputLabelImageList = glob.glob(inputImageLabelsDir + '*.PNG')

    #Check all name in input list and try to find same name +'_label' in label list
    #If name doesn't exist, create empty black image with same dimension
    for imagePath in inputImageList:
        inputImageNameWithExt = imagePath.rsplit('\\', 1)[1]
        inputImageName, inputImageNameExtension = os.path.splitext(inputImageNameWithExt)
        print('Opening: ' + imagePath)
        inputImageMat = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        imageHeight, imageWidth = inputImageMat.shape
        if inputImageMat.any():
            print('Opened succesfully!')
        else:
            print('Cant open, inspect better...')
            continue
        labelFound = False
        for imageLabelPath in inputLabelImageList:
            inputLabelImageNameWithExt = imageLabelPath.rsplit('\\', 1)[1]
            inputLabelImageName, inputLabelImageNameExtension = os.path.splitext(inputLabelImageNameWithExt)
            #check if image name is in label name
            if inputImageName in imageLabelPath:
                labelFound = True
        #If nothing was found, just make and label image, all black
        if not labelFound:
            blackImage = np.zeros((imageWidth, imageHeight, 1), np.uint8)
            newLabelImageSavingPath = outputImageLabelsDir + inputImageName + '_label.PNG'
            cv2.imwrite(newLabelImageSavingPath, blackImage)
            print('Image saved in ' + newLabelImageSavingPath)
    #THE END

    print('DONE!')
    
    

        



