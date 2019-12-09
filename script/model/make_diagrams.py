import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from script.processing.DataAnalysis.benchmark import Benchmark
from script.processing.DataAnalysis.statistics import Statistics
from script.processing.DataAnalysis.imageData import ImageData

def makeGraph(x, y):
    print('Making diagram..')

    y_pos = np.arange(len(y))

    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x, rotation='vertical')
    plt.ylabel('Dice Score')
    plt.title('Best Dice scores')
    plt.tight_layout()
    # function to show the plot
    plt.show()


def main():
    architecturesInputDir = 'D:/CracksTrainings/Set_1/'
    # Get subdirectories of all architectures
    inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
    x = []
    y = []
    for inputArchitectureSurDir in inputArchitecturesSubDirs:
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
        x.append(dir_name)
        score, epoch = currentBenchmark.GetBestDice()
        y.append(score)
        input_file.close()
    makeGraph(x, y)


if __name__ == "__main__":
    main()