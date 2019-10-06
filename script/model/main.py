from model import *
from data import *
import time

"""
#make iteration throught every class data


data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.00,
                    horizontal_flip=False,
                    fill_mode='nearest')

dataDir = 'C:/src/DAGM/'
outputDir = 'C:/src/DAGM/5k32/'
for i in range(1,7):
    print('Sleep for 30s !')
    time.sleep(30)

    currentInputDir = dataDir + 'class' + str(i) + '/'
    currentOutputDir = outputDir + 'class' + str(i) + '/'
    if not os.path.exists(currentOutputDir):
        print('Output results directory doesnt exist!\n')
        print('It will be created!\n')
        os.makedirs(currentOutputDir)

    myGene = trainGenerator(2,'C:/src/DAGM/Augmented4/','image','label',data_gen_args,save_to_dir = None)

    model = unet()
    outputPath = "unet_dagm4-{epoch:03d}-{loss:.4f}.hdf5"
    model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
    model.fit_generator(myGene,steps_per_epoch=575,epochs=100,callbacks=[model_checkpoint])

    print('Sleep for 300s !')
    time.sleep(300)

#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)
"""

model = unet_2layerWithoutBatchNormStride2()