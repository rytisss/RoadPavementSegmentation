import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def makeGraph(labels, y1, y2, name, output='', plot_width=9, plot_height=6, ymin = 0.0, ymax = 1.0):
    # just print all architecture names
    print('Making diagram..')

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    rects1 = ax.bar(x - width / 2, y1, width, label='Pretrained', edgecolor = '#091229')
    rects2 = ax.bar(x + width / 2, y2, width, label='Additional Training', edgecolor = '#091229')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Dice Score', fontsize=16)
    ax.set_xlabel('Model title', fontsize=16)
    ax.set_title(name, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.set_xticklabels(labels, rotation='vertical')
    ax.legend(fontsize=14)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.4)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.2)
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            textOffset = (ymax - ymin) / 20.0
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, ymin + textOffset),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=20)

    autolabel(rects1)
    autolabel(rects2)
    plt.ylim((ymin, ymax))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()

    #plt.show()

    # function to show the plot
    # plt.show()
    fig.savefig(output, dpi=400)
    plt.close()

def main():
    labels = ['UNet','ResUNet','ResUNet+\nASPP','ResUNet+\nASPP+\nAG','ResUNet+\nASPP_WF','ResUNet+\nASPP_WF+\nAG']
    crackForestValues = [0.7015, 0.7058, 0.7121, 0.6969, 0.7114, 0.7106]
    crackForestMixValues = [0.6885, 0.6973, 0.6608, 0.6840, 0.6409, 0.6774]
    crack500Values = [0.6803, 0.6819, 0.6820, 0.6893, 0.6931, 0.6882]
    crack500MixValues = [0.6767, 0.6819, 0.6744, 0.6768, 0.6821, 0.6864]
    gaps384Values = [0.5448, 0.557, 0.5786, 0.5822, 0.5696, 0.5694]
    gaps384MixValues = [0.4733, 0.5107, 0.4588, 0.5185, 0.5321, 0.5068]

    #base directory for output
    output = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/diagrams/'

    makeGraph(labels, crackForestMixValues, crackForestValues, 'Dice Scores on CrackForest Dataset',
              output + 'crackForest_dice.png', 9, 6, 0.59, 0.75)
    makeGraph(labels, crack500MixValues, crack500Values, 'Dice Scores on Crack500 Dataset',
              output + 'crack500_dice.png', 9, 6, 0.65, 0.705)
    makeGraph(labels, gaps384MixValues, gaps384Values, 'Dice Scores on GAPs384 Dataset',
              output + 'gaps384_dice.png', 9, 6, 0.39, 0.62)

if __name__ == "__main__":
    main()