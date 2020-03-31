import glob
import os

class ImageData:
    def __init__(self):
        imagePath = ''
        labelPath = ''
        predictionPath = ''
        imageFiles = []
        labelFiles = []
        predictionFiles = []
    
    def load(self, imagePath, labelPath, predictionPath):
        self.imagePath = imagePath
        self.labelPath = labelPath
        self.predictionPath = predictionPath

        self.imageFiles = glob.glob(imagePath + '*.png')
        self.labelFiles = glob.glob(labelPath + '*.png')
        self.predictionFiles = glob.glob(predictionPath + '*.png')

        # Prediction files don't have zero padding, so we need to sort then in order
        # They should represent actual image
        self.predictionFiles = self.sortPrediction(self.predictionFiles)

    def sortPrediction(self, predictionFiles):
        extension = '_predict'
        sortedList = []
        #indexes start with 0, search for first then iterate
        for i in range(len(predictionFiles)):
            for predictionFile in predictionFiles:
                fileNameWithExt = predictionFile.rsplit('\\', 1)[1]
                fileName, fileExtension = os.path.splitext(fileNameWithExt)
                #take out extension
                extractedName = fileName.replace(extension, '')
                #parse to number
                index = int(extractedName)
                if i == index:
                    sortedList.append(predictionFile)
                    break
        return sortedList

    def getDataCount(self):
        return len(self.predictionFiles)

    def getImageData(self, index):
        imagePath = self.imageFiles[index]
        labelPath = self.labelFiles[index]
        predictionPath = self.predictionFiles[index]
        return imagePath, labelPath, predictionPath



        
