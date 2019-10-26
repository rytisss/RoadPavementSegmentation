import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmark import Benchmark
from statistics import Statistics
from render import Render
from imageData import ImageData

def makeGraph(Benchmark benchmark):


def main():
        architecturesInputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/'
        #Get subdirectories of all architectures
        inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
        
        for inputArchitectureSurDir in inputArchitecturesSubDirs:
                txtPath = inputArchitectureSurDir + 'averageScore.txt'
                inputFile = open(txtPath,'r')
                currentBenchmark = Benchmark()
                count = 0
                for lineText in inputFile:
                        #skip first line
                        if count == 0:
                                count = count + 1
                                continue
                        currentBenchmark.parseDataLine(lineText, count)
                        makeGraph(currentBenchmark)
                inputFile.close()


if __name__== "__main__":
        main()

