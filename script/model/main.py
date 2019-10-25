from model import *
import time
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2

#make iteration throught every class data

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
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
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
        shuffle=True)
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
        shuffle=True)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)

        showNetworkData = False
        if showNetworkData:
            #testing purpose
            mat1 = img[0,:,:,:]
            mat1 *= 255
            mat1 = mat1.astype(np.uint8)
            mat2 = img[1,:,:,:]
            mat2 *= 255
            mat2 = mat2.astype(np.uint8)
            mask1 = mask[0,:,:,:]
            mask1 *= 255
            mask1 = mask1.astype(np.uint8)
            mask2 = mask[1,:,:,:]
            mask2 *= 255
            mask2 = mask2.astype(np.uint8)
        
            cv2.imshow('mask1', mask1)
            cv2.imshow('mask2', mask2)
            cv2.imshow('mat1', mat1)
            cv2.imshow('mat2', mat2)
            cv2.waitKey(0)

        yield (img,mask)

data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.00,
                    height_shift_range=0.00,
                    shear_range=0.00,
                    zoom_range=0.00,
                    horizontal_flip=False,
                    fill_mode='nearest')

#1
outputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/l5k16ResAddOPConcResDice/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(2,'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/','Images','Labels',data_gen_args,save_to_dir = None)

model = AutoEncoder5ResAddOpConcDec(loss_function = Loss.DICE)
outputPath = outputDir + "AutoEncoder5ResAddOpConcResDice-{epoch:03d}-{loss:.4f}.hdf5"
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(generator,steps_per_epoch=176,epochs=50,callbacks=[model_checkpoint])

#2
outputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/l5k16ResAddOPConcResCross/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(2,'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/','Images','Labels',data_gen_args,save_to_dir = None)

model = AutoEncoder5ResAddOpConcDec(loss_function = Loss.CROSSENTROPY)
outputPath = outputDir + "AutoEncoder5ResAddOpConcResCross-{epoch:03d}-{loss:.4f}.hdf5"
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(generator,steps_per_epoch=176,epochs=50,callbacks=[model_checkpoint])

#3
outputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/l5k16ResAddOPConcResDice_1/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(2,'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/','Images','Labels',data_gen_args,save_to_dir = None)

model = AutoEncoder5ResAddOpConcDec(loss_function = Loss.DICE)
outputPath = outputDir + "AutoEncoder5ResAddOpConcResDice-{epoch:03d}-{loss:.4f}.hdf5"
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(generator,steps_per_epoch=176,epochs=50,callbacks=[model_checkpoint])

#4
outputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/l5k16ResAddOPConcResCross_1/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(2,'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/','Images','Labels',data_gen_args,save_to_dir = None)

model = AutoEncoder5ResAddOpConcDec(loss_function = Loss.CROSSENTROPY)
outputPath = outputDir + "AutoEncoder5ResAddOpConcResCross-{epoch:03d}-{loss:.4f}.hdf5"
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(generator,steps_per_epoch=176,epochs=50,callbacks=[model_checkpoint])

#5
outputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/l5k16ResAddOPConcResDice_2/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(2,'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/','Images','Labels',data_gen_args,save_to_dir = None)

model = AutoEncoder5ResAddOpConcDec(loss_function = Loss.DICE)
outputPath = outputDir + "AutoEncoder5ResAddOpConcResDice-{epoch:03d}-{loss:.4f}.hdf5"
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(generator,steps_per_epoch=176,epochs=50,callbacks=[model_checkpoint])

#6
outputDir = 'C:/Users/DeepLearningRig/Desktop/trainingOutput_new/l5k16ResAddOPConcResCross_2/'
if not os.path.exists(outputDir):
    print('Output directory doesnt exist!\n')
    print('It will be created!\n')
    os.makedirs(outputDir)

generator = trainGenerator(2,'C:/Users/DeepLearningRig/Desktop/crackForestDataset/SeparatedDataset/Set_0/Train/Augm/','Images','Labels',data_gen_args,save_to_dir = None)

model = AutoEncoder5ResAddOpConcDec(loss_function = Loss.CROSSENTROPY)
outputPath = outputDir + "AutoEncoder5ResAddOpConcResCross-{epoch:03d}-{loss:.4f}.hdf5"
model_checkpoint = ModelCheckpoint(outputPath, monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(generator,steps_per_epoch=176,epochs=50,callbacks=[model_checkpoint])