import os
import glob
import cv2
import numpy as np

def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list

def CheckIfPixelInSpecificPlaceWithinTolerance(x,y,label,prediction,tolerance):
    height, width = label.shape
    #basically take square with offset 'tolerance'. Also check if we are not out of image range
    #cv2.imshow('predictionTest', prediction)
    #cv2.waitKey(1)
    #value of a particular pixel in label
    label_val = label[y, x]
    #store all value in range
    pixels_values = []
    left = x - tolerance
    right = x + tolerance + 1 # +1, cause last index in range is not evaluated
    top = y - tolerance
    bottom = y + tolerance + 1 #+1, cause last index in range is not evaluated

    visualize = False
    if visualize:
        visualize_matrix = np.zeros((height, width), np.uint8)

    #if visualize:
        #print('Current pixel: ' + str(x) + ', ' + str(y))
    counter = 0
    for x_ in range(left, right):
        for y_ in range(top, bottom):
            #check if pixel is within image
            if x_ >= 0 and x_ < width and y_ >= 0 and y_ < height:

                prediction_val = prediction[y_, x_]
                if prediction_val != 0 and prediction_val != 255:
                    print('Something is wrong!')
                if visualize:
                    #print(str(x_) + ', ' + str(y_))
                    visualize_matrix[y_, x_] = 255
                    cv2.imshow('visual', visualize_matrix)
                    cv2.waitKey(1)
                if prediction_val == label_val:
                    #print(counter)
                    return True #found
            else:
                h = 4
            counter += 1
    #cv2.waitKey(1000)
    return False #nothing is found

def AnalyzeSample(image, label, prediction):
    height, width = image.shape
    #tolerance in pixel how far another 'possible' pixel can be
    tolerance = 2
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for y in range(0, height):
        for x in range(0, width):
            res = CheckIfPixelInSpecificPlaceWithinTolerance(x,y,label,prediction,tolerance)
            # check the value to know what what we are looking for: positives or negatives
            if label[y, x] == 0:
                #negative
                if res:
                    tn += 1
                else:
                    fn += 1
            else:
                if res:
                    tp += 1
                else:
                    fp += 1
    #print('True positive: ' + str(tp))
    #print('False positive: ' + str(fp))
    #print('True negative: ' + str(tn))
    #print('False negative: ' + str(fn))
    return tp, tn, fp, fn

def GetParameters(tp, tn, fp, fn):
    acc = (float)(tp + tn) / (float)(tp + tn + fp + fn)
    recall = (float)(tp) / (float)(tp + fn)
    precision = (float)(tp) / (float)(tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Accuracy: ' + str(acc) + ', Recall: ' + str(recall) + ', Precision: ' + str(precision) + ', F1: ' + str(f1))

def AnalyzePredictions():
    #gather all test image and labels
    path_to_test_data = r'D:\pavement defect data\crack500_out_0.25percent_size\Test/'
    prediction_paths = r'C:\Users\Rytis\Desktop\pavement_defect_results\pretrained_UNet4\crack500\Crack500_pretrained_UNet4_32_5850/'
    imagePaths = gather_image_from_dir(path_to_test_data + 'Images/')
    labelPaths = gather_image_from_dir(path_to_test_data + 'Labels/')
    predictionImagePaths = gather_image_from_dir(prediction_paths)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, len(predictionImagePaths)):
        image = cv2.imread(imagePaths[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(labelPaths[i], cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(predictionImagePaths[i], cv2.IMREAD_GRAYSCALE)
        _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
        _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('image', image)
        cv2.imshow('label', label)
        cv2.imshow('prediction', prediction)
        cv2.waitKey(1)
        tp_image, tn_image, fp_image, fn_image = AnalyzeSample(image, label, prediction)
        GetParameters(tp_image, tn_image, fp_image, fn_image)
        tp += tp_image
        tn += tn_image
        fp += fp_image
        fn += fn_image
    return tp, tn, fp, fn

def AnalyzeArchitecture():
    # Get subdirectories from prediction images
    tp, tn, fp, fn = AnalyzePredictions()
    print('CrackForest' + ',' + str(tp) + ',' + str(tn) + ','+ str(fp) + ','+ str(fn))


def main():
    AnalyzeArchitecture()


if __name__ == "__main__":
    main()





