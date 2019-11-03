import os
import glob
import cv2
import numpy as np
from augmentationTool import AugmentationTool
import random

def saveImage(image, outputPath, name, prefix):
    cv2.imwrite(outputPath + name + prefix + '.bmp', image)

def main():
    inputDir = 'C:/Users/DeepLearningRig/Desktop/datasets/Set_4/Train/'
    imageDir = inputDir + 'Images/'
    labelDir = inputDir + 'Labels/'

    ouputDir = 'C:/Users/DeepLearningRig/Desktop/datasets/Set_4/Train/AUGM/'
    outputImageDir = ouputDir + 'Images/'
    outputLabelDir = ouputDir + 'Labels/'

    if not os.path.exists(outputImageDir):
        os.makedirs(outputImageDir)

    if not os.path.exists(outputLabelDir):
            os.makedirs(outputLabelDir)

    seed = 115
    imagePaths  = glob.glob(imageDir + '*.bmp')
    labelPaths = glob.glob(labelDir + '*.bmp')
    random.Random(seed).shuffle(imagePaths)
    random.Random(seed).shuffle(labelPaths)

    #add additional number to front to be 'shuffle' randomly in directory
    counter = 0

    for i in range(0, len(imagePaths)):
        print("Image: " + imagePaths[i])
        print("Label: " + labelPaths[i])
        imageNameWithExt = imagePaths[i].rsplit('\\', 1)[1]
        imageName, imageExtension = os.path.splitext(imageNameWithExt)

        labelNameWithExt = labelPaths[i].rsplit('\\', 1)[1]
        labelName, labelExtension = os.path.splitext(imageNameWithExt)

        image = cv2.imread(imagePaths[i], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(labelPaths[i], cv2.IMREAD_UNCHANGED)

        #rotation 0
        frontName = str(random.randint(0,1000)) + '_'
        print("Random: " + frontName)
        saveImage(image, outputImageDir, frontName + imageName, '_rot0')
        saveImage(label, outputLabelDir, frontName + labelName, '_rot0')

        #rotation 180
        image180 = AugmentationTool.RotateImage(image, 180)
        label180 = AugmentationTool.RotateImage(label, 180)
        frontName = str(random.randint(0,1000))+ '_'
        print("Random: " + frontName)
        saveImage(image180, outputImageDir, frontName + imageName, '_rot180')
        saveImage(label180, outputLabelDir, frontName + labelName, '_rot180')

        #flip
        imageflip = AugmentationTool.FlipImageHorizontally(image)
        labelflip = AugmentationTool.FlipImageHorizontally(label)
        frontName = str(random.randint(0,1000))+ '_'
        print("Random: " + frontName)
        saveImage(imageflip, outputImageDir, frontName + imageName, '_flip')
        saveImage(labelflip, outputLabelDir, frontName + labelName, '_flip')

        #flip and rotate 180
        imageflip = AugmentationTool.FlipImageHorizontally(image)
        labelflip = AugmentationTool.FlipImageHorizontally(label)
        imagefliprot180 = AugmentationTool.RotateImage(imageflip, 180)
        labelfliprot180 = AugmentationTool.RotateImage(labelflip, 180)
        frontName = str(random.randint(0,1000))+ '_'
        print("Random: " + frontName)
        saveImage(imagefliprot180, outputImageDir, frontName + imageName, '_flip_rot180')
        saveImage(labelfliprot180, outputLabelDir, frontName + labelName, '_flip_rot180')

if __name__ == '__main__':
    main()
        