import os
import glob
import cv2
import numpy as np
import random
import math


def saveImage(image, outputPath, name, prefix):
    cv2.imwrite(outputPath + name + prefix + '.jpg', image)


def splitImageToTiles(width, height, tileWidth, tileHeight, tileOverlayX, tileOverlayY):
    tileRegions = []
    if tileWidth > width or tileHeight > height:
        return tileRegions  # nothing to do in this case if tile is bigger than image

    inWidthRange = True
    inHeightRange = True
    currentX = 0
    currentY = 0
    stepX = tileWidth - tileOverlayX
    stepY = tileHeight - tileOverlayY
    lastIteration = False
    while inHeightRange:
        inWidthRange = True
        currentX = 0
        while inWidthRange:
            # form region
            if currentX >= width - tileWidth - 1:
                currentX = width - tileWidth - 1  # adapt region to fit into image
                inWidthRange = False
            region = (currentX, currentY, currentX + tileWidth, currentY + tileHeight)
            tileRegions.append(region)
            currentX += stepX
        currentY += stepY
        if lastIteration:
            inHeightRange = False
        if currentY >= height - tileHeight - 1:
            currentY = height - tileHeight
            lastIteration = True
        if tileHeight == height:
            break
    return tileRegions


def cropImageFromRegion(image, roi):
    crop_img = image[roi[1]:roi[3], roi[0]:roi[2]]
    return crop_img