import cv2
import numpy as np

class Statistics:
    @staticmethod
    def GetParameters(groundTruth, prediction):
        imageHeight, imageWidth  = groundTruth.shape
        #true positives
        tpMatrix = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_and(groundTruth, prediction, tpMatrix)
        tp = cv2.countNonZero(tpMatrix)
        #false positive
        fpMatrix = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_xor(prediction, tpMatrix, fpMatrix)
        fp = cv2.countNonZero(fpMatrix)
        #false negative
        fnMatrix = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_xor(groundTruth, tpMatrix, fnMatrix)
        fn = cv2.countNonZero(fnMatrix)
        #true negative
        tn = imageHeight * imageWidth - tp - fp - fn
        return tp, fp, tn, fn

    @staticmethod
    def GetRecall(tp, fn):
        divider = tp + fn
        recall = 0.0
        if divider == 0.0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)
        recall = round(recall, 4)
        return recall

    @staticmethod
    def GetPrecision(tp, fp):
        divider = tp + fp
        precision = 0.0
        if divider == 0.0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)
        precision = round(precision, 4)
        return precision

    @staticmethod
    def GetAccuracy(tp, fp, tn, fn):
        divider = tp + fp + tn + fn
        accuracy = 0.0
        if divider == 0.0:
            accuracy = 1.0
        else:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy = round(accuracy, 4)
        return accuracy

    @staticmethod
    def GetF1Score(recall, precision):
        devider = precision + recall
        f1 = 0
        if devider == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (precision + recall)
        f1 = round(f1, 4)
        return f1

    @staticmethod
    def GetIoU(groundTruth, prediction):
        imageHeight, imageWidth  = groundTruth.shape
        
        intersection = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_and(groundTruth, prediction, intersection)

        union = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_or(groundTruth, prediction, union)

        intersectionPixels = cv2.countNonZero(intersection)
        unionPixels = cv2.countNonZero(union)
        iou = 0.0
        if unionPixels == 0:
            iou = 1.0
        else:
            iou = intersectionPixels / unionPixels
        iou = round(iou, 4)
        return iou

    @staticmethod
    def GetDiceCoef(groundTruth, prediction):
        smooth = 1e-6

        imageHeight, imageWidth  = groundTruth.shape
        
        intersection = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_and(groundTruth, prediction, intersection)

        union = np.zeros((imageHeight,imageWidth,1), np.uint8)
        cv2.bitwise_or(groundTruth, prediction, union)

        intersectionPixels = cv2.countNonZero(intersection)
        groundTruthPixels = cv2.countNonZero(groundTruth)
        predictionTruthPixels = cv2.countNonZero(prediction)

        dice = (2. * intersectionPixels + smooth) / (groundTruthPixels + predictionTruthPixels + smooth)
        dice = round(dice, 4)
        return dice

        
   