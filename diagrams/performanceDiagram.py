import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def makeGraphParameters(labels, y1, name, output='', plot_width=9, plot_height=6, ymin = 0.0, ymax = 1.0):
    # just print all architecture names
    print('Making diagram..')

    x = np.arange(len(labels))  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    rects1 = ax.bar(x, y1, width, label='-', edgecolor = '#091229')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of parameters', fontsize=16)
    ax.set_xlabel('Model title', fontsize=16)
    ax.set_title(name, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.set_xticklabels(labels, rotation='vertical')
    #ax.legend(fontsize=14)
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
                        ha='center', va='bottom', rotation=90, fontsize=24)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))
    autolabel(rects1)
    plt.ylim((ymin, ymax))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()

    #plt.show()

    # function to show the plot
    # plt.show()
    fig.savefig(output, dpi=400)
    plt.close()

def makeGraphSeconds(labels, y1, name, output='', plot_width=9, plot_height=6, ymin = 0.0, ymax = 1.0):
    # just print all architecture names
    print('Making diagram..')

    x = np.arange(len(labels))  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    rects1 = ax.bar(x, y1, width, label='-', edgecolor = '#091229', facecolor = '#FF7B0E')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Prediction duration, ms', fontsize=16)
    ax.set_xlabel('Model title', fontsize=16)
    ax.set_title(name, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.set_xticklabels(labels, rotation='vertical')
    #ax.legend(fontsize=14)
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
                        ha='center', va='bottom', rotation=90, fontsize=24)

    autolabel(rects1)
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
    parameters = [1932109, 1997293, 4034029, 4096047, 4034029, 4096047]
    time = [12.94, 13.76, 15.58, 16.21, 15.49, 16.15]


    #base directory for output
    output = 'C:/Users/Rytis/Desktop/Straipsniai/Sensors_FromIDAACS_v2/diagrams/'

    makeGraphParameters(labels, parameters, 'Number of Parameters of Each Model',
              output + 'parameters.png', 9, 6, 0, 4200000)

    makeGraphSeconds(labels, time, '320' + r'$\times$' + '320px Greyscale Image Prediction Performance',
                        output + 'performance.png', 9, 6, 0, 18)

if __name__ == "__main__":
    main()