class Benchmark:
    def __init__(self):
        self.classNr = 0
        self.configName = ''
        self.minAreaFilter = 0
        self.trainingDiceList = []
        self.accuracyList = []
        self.precisionList = []
        self.recallList = []
        self.iouList = []
        self.f1List = []
        self.diceList = []
        self.epochNumbersList = []
        """
    averageScoreLine = str(epochNumber) + ' ' + 
    str(trainingDice) + ' ' + 
    str(overallRecall) + ' ' + 
    str(overallPrecision) + ' ' + 
    str(overallAccuracy) + ' ' + 
    str(overallF1) + ' ' + 
    str(overallIoU) + ' ' +
     str(overallDice) + '\n'
         """
    def parseDataLine(self, line, i):
        words = line.split()
        #epochNumber = int(words[0])
        self.epochNumbersList.append(i)
        recall = float(words[2])
        self.recallList.append(recall)
        precision = float(words[3])
        self.precisionList.append(precision)
        accuracy = float(words[4])
        self.accuracyList.append(accuracy)
        f1 = float(words[5])
        self.f1List.append(f1)
        iou = float(words[6])
        self.iouList.append(iou)
        dice = float(words[7])
        self.diceList.append(dice)

    def Clear(self):
        self.minAreaFilter = 0
        self.trainingDiceList.Clear()
        self.accuracyList.Clear()
        self.precisionList.Clear()
        self.recallList.Clear()
        self.iouList.Clear()
        self.f1List.Clear()
        self.diceList.Clear()
        self.epochNumbersList.clear()
    
    def GetBestAccuracy(self):
        accuracy, epoch = self.GetBiggestParameterAndIndex(self.accuracyList)
        return accuracy, epoch

    def GetBestPrecision(self):
        precision, epoch = self.GetBiggestParameterAndIndex(self.precisionList)
        return precision, epoch

    def GetBestTrainingDice(self):
        trainingDice, epoch = self.GetBiggestParameterAndIndex(self.trainingDiceList)
        return trainingDice, epoch

    def GetBestRecall(self):
        recall, epoch = self.GetBiggestParameterAndIndex(self.recallList)
        return recall, epoch

    def GetBestIoU(self):
        iou, epoch = self.GetBiggestParameterAndIndex(self.iouList)
        return iou, epoch

    def GetBestF1(self):
        f1, epoch = self.GetBiggestParameterAndIndex(self.f1List)
        return f1, epoch

    def GetBestDice(self):
        dice, epoch = self.GetBiggestParameterAndIndex(self.diceList)
        return dice, epoch

    def GetAccuracyAt(self, index):
        return self.GetParameterAt(self.accuracyList, index)

    def GetRecallAt(self, index):
        return self.GetParameterAt(self.recallList, index)

    def GetPrecisionAt(self, index):
        return self.GetParameterAt(self.precisionList, index)

    def GetIoUAt(self, index):
        return self.GetParameterAt(self.iouList, index)

    def GetDiceAt(self, index):
        return self.GetParameterAt(self.diceList, index)

    def GetParameterAt(self, elementList, index):
        return elementList[index]

    def GetBiggestParameterAndIndex(self, elementList):
        biggestElement = 0
        biggestElementIndex = -1
        counter = 0
        for element in elementList:
            if biggestElement < element:
                biggestElementIndex = counter
                biggestElement = element
            counter = counter + 1
        return biggestElement, biggestElementIndex
