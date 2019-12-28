import os
import glob
import cv2
import numpy as np
from script.processing.DataAnalysis.statistics import Statistics
from script.processing.DataAnalysis.render import Render
import matplotlib.pyplot as plt

def get_results_from_order(x,y,order):
    x_sort = []
    y_sort = []
    for i in range(0, len(order)):
        for j in range(0, len(x)):
            order_elem = order[i]
            x_elem = x[j]
            if order_elem in x_elem:
                x_sort.append(x[j])
                #x_sort.append(x[j+1])
                y_sort.append(y[j])
                #y_sort.append(y[j+1])
    return x_sort, y_sort

def get_single_loss_res_ordered(x, y):
    print('Ordering single loss funtion results')
    order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_DICE_0_0.001_'
    ]
    return get_results_from_order(x, y, order)

def get_combo_loss_res_ordered(x, y):
    print('Ordering combo loss funtion results')
    order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_'
    ]
    return get_results_from_order(x, y, order)

def GetCurrentPredictionInfo(path):
    info = os.path.basename(os.path.normpath(path))
    # three first symbols discribes epoch, then '-' and the rest 6 symbols are dice coeficient
    words = info.split('-')
    epochNumber = int(words[1])
    diceLoss = float(words[2])
    return epochNumber, diceLoss

def parse_label(label):
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}^{\mathbf{25\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{75\%}}$'
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}^{\mathbf{50\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{50\%}}$'
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}^{\mathbf{75\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{25\%}}$'
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}$'
    if 'l4k32AutoEncoder4_5x5_DICE_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{D}}$'
    if 'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{DB}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{W60CE}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{W70CE}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{25\%}}$'+'+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{75\%}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{50\%}}$'+'+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{50\%}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{75\%}}$'+'+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{25\%}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}$'

def makeGraph(x, y, name, output='', plot_width=8, plot_height=5):
    y_pos = np.arange(len(y))
    # plt.figure(1, figsize=(12,16))
    labels = []
    for i in range(0, len(x)):
        label = parse_label(x[i])
        labels.append(label)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.4', color='red', alpha=0.30)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.3', color='red', alpha=0.15)

    xlab = np.arange(len(y[0]))  # the label locations

    marker_size = 5
    line_width=1
    # first set
    ax.plot(xlab, y[0], color="#264653", marker='s', markerfacecolor="#264653", markeredgecolor='#1a2f38',
            label=labels[0], markeredgewidth=0.7, markersize=marker_size, linewidth = line_width, alpha=0.8)
    ax.plot(xlab, y[1], color="#2A9D8F", marker='o', markerfacecolor="#2A9D8F", markeredgecolor='#185951',
            label=labels[1], markeredgewidth=0.7, markersize=marker_size, linewidth = line_width, alpha=0.8)
    ax.plot(xlab, y[2], color="#E9C46A", marker='X', markerfacecolor="#E9C46A", markeredgecolor='#3d331a',
            label=labels[2], markeredgewidth=0.7, markersize=marker_size, linewidth = line_width, alpha=0.8)
    ax.plot(xlab, y[3], color="#F4A261", marker='D', markerfacecolor="#F4A261", markeredgecolor='#422c1a',
            label=labels[3], markeredgewidth=0.7, markersize=marker_size, linewidth = line_width, alpha=0.8)
    ax.plot(xlab, y[4], color="#E76F51", marker='^', markerfacecolor="#E76F51", markeredgecolor='#5c2b1f',
            label=labels[4], markeredgewidth=0.7, markersize=marker_size, linewidth = line_width, alpha=0.8)

    ax.set_xticks(xlab)
    ax.set_xticklabels(xlab, rotation='vertical')
    ax.tick_params(axis='x', which='minor', bottom=False)
    plt.ylabel('Dice Score')
    plt.xlabel('Image Sample ID')
    plt.title(name)
    plt.ylim((0.48, 0.85))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    # function to show the plot
    # plt.show()
    fig.savefig(output, dpi=400)
    plt.close()

def AnalyzeArchitecture():
    inputDirs = ['C:/Users/Rytis\Desktop/CrackTrainings_best/Set_3/l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_3/',
                 'C:/Users/Rytis\Desktop/CrackTrainings_best/Set_3/l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_3/',
                 'C:/Users/Rytis\Desktop/CrackTrainings_best/Set_3/l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_3/',
                 'C:/Users/Rytis\Desktop/CrackTrainings_best/Set_3/l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_3/',
                 'C:/Users/Rytis\Desktop/CrackTrainings_best/Set_3/l4k32AutoEncoder4_5x5_DICE_0_0.001_3/']

    all_dice = []

    for inputDir in inputDirs:
        configName = os.path.basename(os.path.normpath(inputDir))
        imagePath = 'D:/RoadCracksInspection/datasets/Set_3/Test/Images/'
        labelsPath = 'D:/RoadCracksInspection/datasets/Set_3/Test/Labels/'
        images = glob.glob(imagePath + '*.bmp')
        labels = glob.glob(labelsPath + '*.bmp')
        predictions = glob.glob(inputDir + '*.bmp')

        dataCount = len(images)

        dice_values = []
        image_paths = []
        label_paths = []
        for i in range(0, len(images)):
            imagePath = images[i]
            labelPath = labels[i]
            predictionPath = predictions[i]
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            prediction = cv2.imread(predictionPath, cv2.IMREAD_GRAYSCALE)
            _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

            # do analysis
            tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
            recall = Statistics.GetRecall(tp, fn)
            precision = Statistics.GetPrecision(tp, fp)
            accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
            f1 = Statistics.GetF1Score(recall, precision)
            IoU = Statistics.GetIoU(label, prediction)
            dice = Statistics.GetDiceCoef(label, prediction)
            # print('Recall: ' + str(recall) + ', Precision: ' + str(precision) + ', accuracy: ' + str(accuracy) + ', f1: ' + str(f1) + ', IoU: ' + str(IoU) + ', Dice: ' + str(dice))
            dice_values.append(dice)
            image_paths.append(imagePath)
            label_paths.append(labelPath)
            # render defect on image
            outerColor = (0, 0, 200)
            innerColor = (0, 0, 80)
            renderedImage = Render.Defects(image, prediction, outerColor, innerColor)
            cv2.imshow('Render', renderedImage)
            cv2.waitKey(1)
        all_dice.append(dice_values)
    output = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS/samplesScoresSingle.png'
    name = 'Set_3 Samples Dice Scores'
    makeGraph(inputDirs, all_dice, name, output, 9, 3)
    print('Done!')

def main():
    AnalyzeArchitecture()


if __name__ == "__main__":
    main()





