import os
import glob
import cv2
import numpy as np
from augmentationTool import AugmentationTool
import random
import math

def saveImage(image, outputPath, name, prefix):
    cv2.imwrite(outputPath + name + prefix + '.bmp', image)

def splitImageToTiles(width, height, tileWidth, tileHeight, tileOverlay):
    tileRegions = []
    if tileWidth > width or tileHeight > height:
        return tileRegions #nothing to do in this case if tile is bigger than image
    
    inWidthRange = True
    inHeightRange = True
    currentX = 0
    currentY = 0
    stepX = tileWidth - tileOverlay
    stepY = tileHeight - tileOverlay
    lastIteration = False
    while inHeightRange:
        inWidthRange = True
        currentX = 0
        while inWidthRange:
            #form region
            if currentX >= width - tileWidth - 1:
                currentX = width - tileWidth - 1#adapt region to fit into image
                inWidthRange = False
            region = (currentX, currentY, currentX + tileWidth, currentY + tileHeight)
            tileRegions.append(region)
            currentX += stepX    
        currentY += stepY
        if lastIteration:
            inHeightRange = False
        if currentY >= height - tileHeight - 1:
            currentY = height - tileHeight - 1
            lastIteration = True
    return tileRegions

def cropImageFromRegion(image, roi):
    crop_img = image[roi[1]:roi[3], roi[0]:roi[2]]
    return crop_img



def main():

    for setIndex in range(0,5):
        inputDir = 'E:/RoadCracksInspection/datasets80-20/Set_' + str(setIndex) + '/Train/'
        imageDir = inputDir + 'Images/'
        labelDir = inputDir + 'Labels/'

        ouputDir = 'E:/RoadCracksInspection/datasets80-20/Set_' + str(setIndex) + '/Train/Smaller/'
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
        tileWidth = 240
        tileHeight = 240
        overlay = 120

        inputWidth = 480
        inputHeight = 320

        regions = splitImageToTiles(inputWidth,inputHeight, tileWidth, tileHeight, overlay)

        for i in range(0, len(imagePaths)):
            print("Image: " + imagePaths[i])
            print("Label: " + labelPaths[i])
            imageNameWithExt = imagePaths[i].rsplit('\\', 1)[1]
            imageName, imageExtension = os.path.splitext(imageNameWithExt)

            labelNameWithExt = labelPaths[i].rsplit('\\', 1)[1]
            labelName, labelExtension = os.path.splitext(imageNameWithExt)

            image = cv2.imread(imagePaths[i], cv2.IMREAD_UNCHANGED)
            label = cv2.imread(labelPaths[i], cv2.IMREAD_UNCHANGED)

            #do cropping, rotation and saving
            regionIndex = 0
            for region in regions:
                
                croppedImage = cropImageFromRegion(image, region)
                croppedLabel = cropImageFromRegion(label, region)

                augment = True
                if not augment:
                    #original
                    frontName = str(random.randint(0,1000)) + '_'
                    print("Random: " + frontName)
                    saveImage(croppedImage, outputImageDir, frontName + imageName, '_')
                    saveImage(croppedLabel, outputLabelDir, frontName + labelName, '_')
                else:
                    flips = [0, 1]
                    rotates = [0,90,180,270]
                    for flip in flips:
                        for rotate in rotates:
                            labelAdd = str(regionIndex) + '_rot_' + str(rotate) + '_flip_' + str(flip)
                            if flip == 1:
                                imageaug = AugmentationTool.FlipImageHorizontally(croppedImage)
                                labelaug = AugmentationTool.FlipImageHorizontally(croppedLabel)
                                imageaug = AugmentationTool.RotateImage(imageaug, rotate)
                                labelaug = AugmentationTool.RotateImage(labelaug, rotate)
                            else:
                                imageaug = AugmentationTool.RotateImage(croppedImage, rotate)
                                labelaug = AugmentationTool.RotateImage(croppedLabel, rotate)
                            frontName = str(random.randint(0,1000)) + '_'
                            print("Random: " + frontName)
                            saveImage(imageaug, outputImageDir, frontName + imageName, labelAdd)
                            saveImage(labelaug, outputLabelDir, frontName + labelName, labelAdd)
                regionIndex+=1
                #rotation 270
                #image270 = AugmentationTool.RotateImage(croppedImage, 270)
                #label270 = AugmentationTool.RotateImage(croppedLabel, 270)
                #frontName = str(random.randint(0,1000))+ '_'
                #print("Random: " + frontName)
                #saveImage(image270, outputImageDir, frontName + imageName, '_rot270')
                #saveImage(label270, outputLabelDir, frontName + labelName, '_rot270')
                

if __name__ == '__main__':
    main()
        