import os
import glob
import cv2
import numpy as np
from augmentationTool import AugmentationTool

def saveImage(image, outputPath, name, prefix):
    cv2.imwrite(outputPath + name + prefix + '.bmp', image)

def main():
    inputDir = 'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/'
    imageDir = inputDir + 'Images/'
    labelDir = inputDir + 'Labels/'

    outputImageDir = 'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/Images/'
    outputLabelDir = 'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/Labels/'

    imagePaths  = glob.glob(imageDir + '*.bmp')
    labelPaths = glob.glob(labelDir + '*.bmp')
    for i in range(0, len(imagePaths)):
        imageNameWithExt = imagePaths[i].rsplit('\\', 1)[1]
        imageName, imageExtension = os.path.splitext(imageNameWithExt)

        labelNameWithExt = labelPaths[i].rsplit('\\', 1)[1]
        labelName, labelExtension = os.path.splitext(imageNameWithExt)

        image = cv2.imread(imagePaths[i], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(labelPaths[i], cv2.IMREAD_UNCHANGED)

        #rotation 0 
        saveImage(image, outputImageDir, imageName, '_rot0')
        saveImage(label, outputLabelDir, labelName, '_rot0')

        #rotation 180
        image180 = AugmentationTool.RotateImage(image, 180)
        label180 = AugmentationTool.RotateImage(label, 180)
        saveImage(image180, outputImageDir, imageName, '_rot180')
        saveImage(label180, outputLabelDir, labelName, '_rot180')

        #flip
        imageflip = AugmentationTool.FlipImageHorizontally(image)
        labelflip = AugmentationTool.FlipImageHorizontally(label)
        saveImage(imageflip, outputImageDir, imageName, '_flip')
        saveImage(labelflip, outputLabelDir, labelName, '_flip')

        #flip and rotate 180
        imageflip = AugmentationTool.FlipImageHorizontally(image)
        labelflip = AugmentationTool.FlipImageHorizontally(label)
        imagefliprot180 = AugmentationTool.RotateImage(imageflip, 180)
        labelfliprot180 = AugmentationTool.RotateImage(labelflip, 180)
        saveImage(imagefliprot180, outputImageDir, imageName, '_flip_rot180')
        saveImage(labelfliprot180, outputLabelDir, labelName, '_flip_rot180')

if __name__ == '__main__':
    main()
        