"""
def MakeVideo(classNr, configuration, predictionPath):
        inputDir = 'C:/Users/rytis/OneDrive/Desktop/Straipsniai/DAGM/Class' + str(classNr) + '/Test/'
        inputImageDir = inputDir + 'image/'
        inputLabelDir = inputDir + 'label/'
        inputPredictionDir = predictionPath

        imageData = ImageData()
        imageData.load(inputImageDir, inputLabelDir, inputPredictionDir)
        
        outputDirectory = 'C:/Users/rytis/OneDrive/Desktop/Straipsniai/Pattern recognition letters/videoRenders/' 
        if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)

        fileName = ""
        recallSum = 0.0
        precisionSum = 0.0
        accuracySum = 0.0
        f1Sum = 0.0
        IoUSum = 0.0
        dicsSum = 0.0

        outputImageWidth = 1536
        outputImageHeight = 384 + 512
        width = 512
        height = 512

        mainTitle = 'Class' + str(classNr) + ' Network-' + configuration

        architectureVideoName = outputDirectory + mainTitle + '.avi'
        architectureOutVideo = cv2.VideoWriter(architectureVideoName,cv2.VideoWriter_fourcc('X','V','I','D'), 30.0, (outputImageWidth,outputImageHeight))

        for i in range(0, imageData.getDataCount()):
                imagePath, labelPath, predictionPath = imageData.getImageData(i)
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)      
                _, label = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
                prediction = cv2.imread(predictionPath, cv2.IMREAD_GRAYSCALE)
                _, prediction = cv2.threshold(prediction,127,255,cv2.THRESH_BINARY)

                #do analysis
                tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
                recall = Statistics.GetRecall(tp, fn)
                precision = Statistics.GetPrecision(tp, fp)
                accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
                f1 = Statistics.GetF1Score(recall, precision)
                IoU = Statistics.GetIoU(label, prediction)
                dice = Statistics.GetDiceCoef(label, prediction)
                #print('Recall: ' + str(recall) + ', Precision: ' + str(precision) + ', accuracy: ' + str(accuracy) + ', f1: ' + str(f1) + ', IoU: ' + str(IoU) + ', Dice: ' + str(dice))

                recallSum = recallSum + recall
                precisionSum = precisionSum + precision
                accuracySum = accuracySum + accuracy
                f1Sum = f1Sum + f1
                IoUSum = IoUSum + IoU
                dicsSum = dicsSum + dice

                #render defect on image
                outerColor = (80,80,200)
                innerColor = (50,50,140)
                renderedPrediction = Render.Defects(image, prediction, outerColor, innerColor)
                
                #render label on original image
                outerColor = (0,0,255)
                innerColor = (0,0,0)
                renderedImage = Render.Defects(image, label, outerColor, innerColor)

                #cast greyscale to rgb
                prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)

                #form output
                outputImage = np.zeros((outputImageHeight, outputImageWidth,3), np.uint8)

                #image offset from top
                imageOffsetFromTop = 256
                textOffset = 15

                imagebottom = imageOffsetFromTop + height
                outputImage[imageOffsetFromTop:imagebottom, 0:width] = renderedImage
                outputImage[imageOffsetFromTop:imagebottom, width:width*2] = prediction
                outputImage[imageOffsetFromTop:imagebottom, width*2:width*3] = renderedPrediction
                #draw rectange to indentify images boarders
                color = (128, 10, 170)
                cv2.rectangle(outputImage, (0, imageOffsetFromTop), (width, height + imageOffsetFromTop), color, 2)
                cv2.rectangle(outputImage, (width, imageOffsetFromTop), (width * 2, height + imageOffsetFromTop), color, 2)
                cv2.rectangle(outputImage, (width * 2, imageOffsetFromTop), (width * 3, height + imageOffsetFromTop), color, 2)

                #render names for images
                resultsTextThickness = 1
                resultsTextScale = 1.1
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontColor  = (255,255,255)
                resultsTextThickness = 2
                lineType  = cv2.LINE_AA
                #render results under
                currentResultsTitle = 'Image + Label'
                textPos = (120, imageOffsetFromTop - textOffset)
                cv2.putText(outputImage, currentResultsTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                currentResultsTitle = 'Prediction'
                textPos = (682, imageOffsetFromTop - textOffset)
                cv2.putText(outputImage, currentResultsTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                currentResultsTitle = 'Image + Prediction'
                textPos = (1110, imageOffsetFromTop - textOffset)
                cv2.putText(outputImage, currentResultsTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                #render main title
                resultsTextScale = 1.8
                mainTitle = 'Class' + str(classNr) + ' Network-' + configuration
                textPos = (465, 150 - textOffset)
                cv2.putText(outputImage, mainTitle, 
                        textPos, 
                        font, 
                        resultsTextScale,
                        fontColor,
                        resultsTextThickness,
                        lineType)

                #make it last for 0.5 second
                for i in range(0, 15):
                        architectureOutVideo.write(outputImage)
"""
import os
import glob
import shutil
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from script.processing.DataAnalysis.benchmark import Benchmark
from script.processing.DataAnalysis.render import Render
from script.processing.DataAnalysis.statistics import Statistics
from script.processing.DataAnalysis.imageData import ImageData


def translate_name(label):
        if 'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_' in label:
                return 'L_CE25%_D75%'
        if 'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_' in label:
                return 'L_CE50%_D50%'
        if 'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_' in label:
                return 'L_CE75%_D25%'
        if 'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_' in label:
                return 'L_CE'
        if 'l4k32AutoEncoder4_5x5_DICE_0_0.001_' in label:
                return 'L_D'
        if 'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_' in label:
                return 'L_DB'
        if 'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_' in label:
                return 'L_W60CE'
        if 'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_' in label:
                return 'L_W70CE'
        if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_' in label:
                return 'L_WCE25%_D75%'
        if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_' in label:
                return 'L_WCE50%_D50%'
        if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_' in label:
                return 'L_WCE75%_D25%'
        if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_' in label:
                return 'L_WCE'

def get_image_width_n_height(image):
    width = image.shape[0]
    height = image.shape[1]
    return width, height

def draw_frame(image):
    width, height = get_image_width_n_height(image)
    borderColor = (180, 52, 235)
    cv2.rectangle(image, (0, 0), (height - 1, width - 1), borderColor, 1)
    return image

def render_video(architecture_dir, images_dir, label_dir, prediction_dir, output_video, set_number):
    image_paths = glob.glob(images_dir + '*.bmp')
    label_paths = glob.glob(label_dir + '*.bmp')
    prediction_paths = glob.glob(prediction_dir + '*.bmp')
    last_dir_folder = translate_name(os.path.basename(os.path.normpath(architecture_dir)))
    title = str(set_number) + ' ' + last_dir_folder
    print(title)

    width = 480 * 4
    bottom_offset = 200
    top_offset = 300
    height = 320 + bottom_offset + top_offset

    architectureVideoName = output_video + title + '.avi'
    architectureOutVideo = cv2.VideoWriter(architectureVideoName, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0,
                                           (width, height))

    for i in range(0, len(image_paths)):
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_paths[i], cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(prediction_paths[i], cv2.IMREAD_GRAYSCALE)
        _, prediction_th = cv2.threshold(prediction,127,255,cv2.THRESH_BINARY)
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        label_color = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        prediction_color = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
        prediction_th_color = cv2.cvtColor(prediction_th, cv2.COLOR_GRAY2RGB)
        outer_color = (0, 0, 200)
        inner_color = (0, 0, 80)
        image_with_prediction = Render.Defects(image, prediction_th, outer_color, inner_color)
        image_with_label = Render.Defects(image, label, outer_color, inner_color)

        #draw frame for all used images
        image_with_label = draw_frame(image_with_label)
        label_color = draw_frame(label_color)
        prediction_color = draw_frame(prediction_color)
        image_with_prediction = draw_frame(image_with_prediction)

        image_height, image_width = get_image_width_n_height(image_with_prediction)

        # form output
        outputImage = np.zeros((height, width, 3), np.uint8)
        #set images
        outputImage[top_offset:top_offset + image_height, 0:image_width] = image_with_label
        outputImage[top_offset:top_offset + image_height, image_width:image_width * 2] = label_color
        outputImage[top_offset:top_offset + image_height, image_width*2:image_width * 3] = prediction_color
        outputImage[top_offset:top_offset + image_height, image_width*3:image_width * 4] = image_with_prediction

        # render names for images
        resultsTextThickness = 1
        resultsTextScale = 1.1
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (255, 255, 255)
        resultsTextThickness = 2
        lineType = cv2.LINE_AA
        text_offset = 20

        # render results under
        currentResultsTitle = 'Image + Label'
        textPos = (120, top_offset - text_offset)
        cv2.putText(outputImage, currentResultsTitle,
                    textPos,
                    font,
                    resultsTextScale,
                    fontColor,
                    resultsTextThickness,
                    lineType)

        currentResultsTitle = 'Label'
        textPos = (image_width + 180, top_offset - text_offset)
        cv2.putText(outputImage, currentResultsTitle,
                    textPos,
                    font,
                    resultsTextScale,
                    fontColor,
                    resultsTextThickness,
                    lineType)

        currentResultsTitle = 'Full Prediction'
        textPos = (2 * image_width + 120, top_offset - text_offset)
        cv2.putText(outputImage, currentResultsTitle,
                    textPos,
                    font,
                    resultsTextScale,
                    fontColor,
                    resultsTextThickness,
                    lineType)

        currentResultsTitle = 'Image + Prediction >50%'
        textPos = (3 * image_width + 10, top_offset - text_offset)
        cv2.putText(outputImage, currentResultsTitle,
                    textPos,
                    font,
                    resultsTextScale,
                    fontColor,
                    resultsTextThickness,
                    lineType)

        # render main title
        resultsTextScale = 1.7
        resultsTextThickness = 2
        title_text_width = cv2.getTextSize(title, font, resultsTextScale, resultsTextThickness)
        title_x_pos = (int)(width / 2 - title_text_width[0][0] / 2)
        textPos = (title_x_pos, top_offset - 100)
        cv2.putText(outputImage, title,
                    textPos,
                    font,
                    resultsTextScale,
                    fontColor,
                    resultsTextThickness,
                    lineType)

        for i in range(0, 15):
            architectureOutVideo.write(outputImage)

        cv2.imshow('output', outputImage)
        cv2.imshow('image', image)
        cv2.imshow('label', label)
        cv2.imshow('prediction', prediction)
        cv2.imshow('prediction50%', prediction_th)
        cv2.imshow('image with prediction', image_with_prediction)
        cv2.imshow('image with label', image_with_label)
        cv2.waitKey(1)

    architectureOutVideo.release()
def main():
    sets = ['Set_0', 'Set_1', 'Set_2', 'Set_3', 'Set_4']
    all_lines = []
    for set in sets:
        set_lines = []
        print('\n' + set + '\n')
        architecturesInputDir = 'D:/CracksTrainings/' + set + '/'
        outputDir = 'D:/CrackTraining_best_video/'
        image_path = 'C:/Users/Rytis/Desktop/datasets/' + set + '/Test/Images/'
        label_path = 'C:/Users/Rytis/Desktop/datasets/' + set + '/Test/Labels/'

        # Get subdirectories of all architectures
        inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
        directory_names = []
        full_paths = []
        for inputArchitectureSurDir in inputArchitecturesSubDirs:
            if 'SEQ' in inputArchitectureSurDir:
                continue #skip
            txt_path = inputArchitectureSurDir + 'averageScore.txt'
            dir_name = os.path.basename(os.path.normpath(inputArchitectureSurDir))
            input_file = open(txt_path, 'r')
            currentBenchmark = Benchmark()
            count = 0
            for lineText in input_file:
                # skip first line
                if count == 0:
                    count = count + 1
                    continue
                currentBenchmark.parseDataLine(lineText, count-1)
                count += 1
            directory_names.append(dir_name)
            score, epoch = currentBenchmark.GetBestDice()
            print('Best dice score' + str(score))
            #to check
            receivedDice = currentBenchmark.GetDiceAt(epoch)
            print('Received dice score' + str(receivedDice))
            print('acc rec precision iou dice')
            acc = currentBenchmark.GetAccuracyAt(epoch)
            rec = currentBenchmark.GetRecallAt(epoch)
            pre = currentBenchmark.GetPrecisionAt(epoch)
            iou = currentBenchmark.GetIoUAt(epoch)
            dice = currentBenchmark.GetDiceAt(epoch)
            line = str(acc) + ',' + str(rec) + ',' + str(pre) + ',' + str(iou) + ',' + str(dice)
            set_lines.append(line)

            #add 1 cause epoch starts to count from 1
            full_path = inputArchitectureSurDir + 'prediction//' + str(epoch) + '//'
            render_video(inputArchitectureSurDir, image_path, label_path, full_path, outputDir, set)

            input_file.close()


if __name__ == "__main__":
    main()


