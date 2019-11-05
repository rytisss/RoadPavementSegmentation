from model import *
import glob
import time
import keras
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator
import cv2

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (320,480),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        shuffle=False)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        shuffle=False)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img *= 255.0
        img = img.astype(np.uint8)
        #io.imsave(os.path.join(save_path,"%d_predict.bmp"%i),img)
        cv2.imwrite(os.path.join(save_path,"%03d_predict.bmp"%i),img)


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def testGenerator(test_path,num_image = 30,target_size = (320,480),flag_multi_class = False,as_gray = True):
    inputImageList = glob.glob(test_path + '*.bmp')
    for i in inputImageList:
        img = io.imread(i,as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.00,
                    horizontal_flip=False,
                    fill_mode='nearest')

configs = [
'l4k16AutoEncoder4ResAddOpConcDecFirstExDice_0.001_1',
'l4k16AutoEncoder4ResAddOpFirstExDice_0.001_1']

for config in configs:
    #configName = 'l5k16Dice_1'
    configName = config
    inputDir = 'E:/RoadCracksInspection/trainingOutput/1/'+configName+'/'
    weightList = glob.glob(inputDir + '*.hdf5')
    counter = 0
    for weightPath in weightList:
        print('Opening: ' + weightPath)
        fileNameWithExt = weightPath.rsplit('\\', 1)[1]
        fileName, extension = os.path.splitext(fileNameWithExt)
        kernels_list = [16,32]
        for kernels in kernels_list:
            try:
                model = AutoEncoder4ResAddOpConcDecFirstEx(number_of_kernels = kernels, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                break
            except:
                print('Not AutoEncoder4ResAddOpConcDecFirstEx')

            try:
                model = AutoEncoder4ResAddOpFirstEx(number_of_kernels = kernels, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                break
            except:
                print('Not AutoEncoder4ResAddOpFirstEx')

            try:
                model = AutoEncoder4(number_of_kernels = kernels, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                break
            except:
                print('Not AutoEncoder4')

            try:
               model = AutoEncoder4ResAddOp(number_of_kernels = kernels, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
               break
            except:
                print('Not AutoEncoder4ResAddOp')

            try:
                model = AutoEncoder4ResAddOpConcDec(number_of_kernels = kernels, pretrained_weights = weightPath, loss_function = Loss.CROSSENTROPY)
                break
            except:
                print('Not AutoEncoder4ResAddOpConcDec')

           

            

        testGene = testGenerator('E:/RoadCracksInspection/datasets/Set_1/Test/Images/')
        results = model.predict_generator(testGene,35,verbose=1)
                
        predictionOutputDir = 'E:/RoadCracksInspection/trainingOutput/1/'+configName+'/prediction/' + str(counter) + '/'
        if not os.path.exists(predictionOutputDir):
            os.makedirs(predictionOutputDir)
        saveResult(predictionOutputDir,results)
        counter+=1
        keras.backend.clear_session()
        #print('Sleep for 5s !')
        #time.sleep(5)