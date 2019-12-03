from script.model.utilities import *
import glob
from script.model.autoencoder import *
import os
import keras as K

data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.00,
                    horizontal_flip=False,
                    fill_mode='nearest')

for setNumber in range(1, 5):
    configs = ['l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_',
               'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_',
               'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_',
               'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_',
               'l4k32AutoEncoder4_5x5_DICE_0_0.001_',
               'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_',
               'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_',
               'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_',
               'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_',
               'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_',
               'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_']
    configNumber = 1
    for config in configs:
        #configName = 'l5k16Dice_1'
        configName = config
        configName += str(setNumber)
        inputDir = 'C:/src/Set_' + str(setNumber) +'/'+ configName+'/'
        weightList = glob.glob(inputDir + '*.hdf5')
        counter = 0
        for weightPath in weightList:
            print('Opening: ' + weightPath)
            fileNameWithExt = weightPath.rsplit('\\', 1)[1]
            fileName, extension = os.path.splitext(fileNameWithExt)
            kernels_list = [32]
            size = (320,480,1)
            for kernels in kernels_list:
                if configNumber == 0:
                    try:
                        model = AutoEncoder4VGG16_5x5(number_of_kernels = kernels,input_size = size, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                        testGene = testGenerator('E:/RoadCracksInspection/datasets/Set_' + str(setNumber) + '/Test/Images/', target_size = (320,480))
                        results = model.predict_generator(testGene,35,verbose=1)      
                        predictionOutputDir = 'E:/RoadCracksInspection/trainingOutput/Set_' + str(setNumber) +'/'+configName+'/prediction/' + str(counter) + '/'
                        if not os.path.exists(predictionOutputDir):
                            os.makedirs(predictionOutputDir)
                        saveResult(predictionOutputDir,results)
                        break
                    except:
                        print('Not AutoEncoder4VGG16_5x5')
                if configNumber == 1:
                    try:
                        model = AutoEncoder4_5x5(number_of_kernels = kernels,input_size = size, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                        testGene = testGenerator('E:/RoadCracksInspection/datasets/Set_' + str(setNumber) + '/Test/Images/', target_size = (320,480))
                        results = model.predict_generator(testGene,35,verbose=1)      
                        predictionOutputDir = 'C:/src/Set_' + str(setNumber) +'/'+configName+'/prediction/' + str(counter) + '/'
                        if not os.path.exists(predictionOutputDir):
                            os.makedirs(predictionOutputDir)
                        saveResult(predictionOutputDir,results)
                        break
                    except:
                        print('Not AutoEncoder4_5x5')
                if configNumber == 2:
                    try:
                        model = AutoEncoder4ResAddOpConcDecFirstEx_5x5(number_of_kernels = kernels,input_size = size, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                        testGene = testGenerator('E:/RoadCracksInspection/datasets/Set_' + str(setNumber) + '/Test/Images/', target_size = (320,480))
                        results = model.predict_generator(testGene,35,verbose=1)      
                        predictionOutputDir = 'E:/RoadCracksInspection/trainingOutput/Set_' + str(setNumber) +'/'+configName+'/prediction/' + str(counter) + '/'
                        if not os.path.exists(predictionOutputDir):
                            os.makedirs(predictionOutputDir)
                        saveResult(predictionOutputDir,results)
                        break
                    except:
                        print('Not AutoEncoder4ResAddOpConcDecFirstEx_5x5')
            """    
            predictionOutputDir = 'E:/RoadCracksInspection/trainingOutputPictures/Set_' + str(setNumber) +'/'+configName+'/prediction/' + str(counter) + '/'
            if not os.path.exists(predictionOutputDir):
                os.makedirs(predictionOutputDir)
            saveResult(predictionOutputDir,results)
            """

            counter+=1
            K.backend.clear_session()
        #configNumber+=1
            #print('Sleep for 5s !')
            #time.sleep(5)