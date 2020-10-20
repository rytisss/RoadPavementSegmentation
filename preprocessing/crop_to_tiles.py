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
    image_width = image.shape[1]
    image_height = image.shape[0]
    # make empty tile and place region that fits into image into this tile
    x = roi[0]
    y = roi[1]
    width = roi[2] - roi[0]
    height = roi[3] - roi[1]
    # fitted roi
    x_ = 0
    y_ = 0
    w_ = 0
    h_ = 0

    crop_image = np.zeros((width, height), dtype=np.uint8)

    if x > image_width:
        return crop_image
    if y > image_height:
        return crop_image
    if x + width < 0:
        return crop_image
    if y + height < 0:
        return crop_image

    #roi starts outside but overlays image (partly or fully)
    if x < 0 and x + width > 0:
        x_ = 0
        width_ = width + x
        #check if 'fittedBoundingBox.width' is not out of image range
        if width_ > image_width:
            width_ = image_width
    #roi starts outside but overlays image (partly or fully)
    if y < 0 and y + height > 0:
        y_ = 0
        height_ = height + y
        #check if 'fittedBoundingBox.height' is not out of image range
        if height_ > image_height:
            height_ = image_height
    #roi start inside image
    if x >= 0 and x < image_width:
        x_ = x
        if x + width <= image_width:
            width_ = width
        else:
            width_ = image_width - x_
    #roi start inside image
    if y >= 0 and y < image_height:
        y_ = y
        if y + height <= image_height:
            height_ = height
        else:
            height_ = image_height - y_
    crop_img_ = image[y_:y_+height_, x_:x_+width_]
    # check how much of the region is out of image and put cropped image it to requested size image
    x_req = 0
    y_req = 0
    if (x < 0):
        x_req = -x
    if (y < 0):
        y_req = -y
    crop_image[y_req:y_req+height_, x_req:x_req+width_] = crop_img_
    return crop_image