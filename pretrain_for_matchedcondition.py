import cv2
import glob
import numpy as np
import os
import sys
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import random
import keras
from keras.models import *
from keras.layers import *
from keras.applications import VGG16
import itertools
from matplotlib.pyplot import imread
from keras import optimizers
from keras import models
from keras.layers.core import Activation, Reshape, Permute
#from keras.layers.convolutional import convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Conv2D
#from keras.layers.normalization import BatchNormalization
import json
import time
from keras import backend as K
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session          
#config = tf.ConfigProto()  
#config.gpu_options.allow_growth = True  
#set_session(tf.Session(config=config)) 
from keras.callbacks import EarlyStopping, ModelCheckpoint


def generate_data(in_dir,out_dir,sub,vid,X_train, Y_train,names,height,width,n_ops):
	#temp_mat = sio.loadmat(mat_dir)	
	mat = sio.loadmat(out_dir+sub+'/'+str(vid)+'.mat')
	vidcap = cv2.VideoCapture(in_dir+sub+'/'+sub+'_'+str(vid)+'.avi')
	success = True
	for i in range(len(mat['masks']['mask1'][0])):
		success,image = vidcap.read()
		X_train.append(image) 
		seg_labels1 = np.zeros((height, width, n_ops), dtype = int)
		seg_labels2 = np.zeros((height, width, n_ops), dtype = int)
		seg_labels3 = np.zeros((height, width, n_ops), dtype = int)
		for j in range(n_ops-1):
			seg_labels1[:,:,j+1] = mat['masks']['mask1'][0][i].astype(int)
			seg_labels2[:,:,j+1] = mat['masks']['mask2'][0][i].astype(int)
			seg_labels3[:,:,j+1] = mat['masks']['mask3'][0][i].astype(int)
		seg_labels1[:,:,0] = ((seg_labels1[:,:,1] == 0)).astype(int)
		seg_labels2[:,:,0] = ((seg_labels2[:,:,1] == 0)).astype(int)
		seg_labels3[:,:,0] = ((seg_labels3[:,:,1] == 0)).astype(int)
		#print(len(y_train))
		#print(np.array(x_train_r).shape)
		#x_train[i,:,:,:] = image
		Y_train[0].append(seg_labels1)
		Y_train[1].append(seg_labels2)
		Y_train[2].append(seg_labels3)
		names.append(sub+'_'+str(vid)+'_'+str(i+1).zfill(3))		
	return X_train, Y_train, names


def ImageSegmentationGenerator(videoloc, matloc, n_ops):
    X_train = []
    Y_train = [[], [], []]
    height=68
    width=68
    
    for n in range(len(videoloc)):
        vidcap = cv2.VideoCapture(videoloc[n])
        mat = sio.loadmat(matloc[n])
        success = True
        for i in range(len(mat['masks_struct']['mask1'][0])):
            success,image = vidcap.read()
            X_train.append(image) 
            seg_labels1 = np.zeros((height, width, n_ops), dtype = int)
            seg_labels2 = np.zeros((height, width, n_ops), dtype = int)
            seg_labels3 = np.zeros((height, width, n_ops), dtype = int)
            for j in range(n_ops-1):
                seg_labels1[:,:,j+1] = mat['masks_struct']['mask1'][0][i].astype(int)
                seg_labels2[:,:,j+1] = mat['masks_struct']['mask2'][0][i].astype(int)
                seg_labels3[:,:,j+1] = mat['masks_struct']['mask3'][0][i].astype(int)
            seg_labels1[:,:,0] = ((seg_labels1[:,:,1] == 0)).astype(int)
            seg_labels2[:,:,0] = ((seg_labels2[:,:,1] == 0)).astype(int)
            seg_labels3[:,:,0] = ((seg_labels3[:,:,1] == 0)).astype(int)
            #print(len(y_train))
            #print(np.array(x_train_r).shape)
            #x_train[i,:,:,:] = image
            Y_train[0].append(seg_labels1)
            Y_train[1].append(seg_labels2)
            Y_train[2].append(seg_labels3)
            
    X_train = np.array(X_train).astype('float64')
    X_train = X_train / 255
    Y_train1 = [np.array(Y_train[i]) for i in range(3)]
    
    return X_train, Y_train1


def predicted_output(model, X_dev, height=68, width=68, n_ops=2):
    batch_len = X_dev.shape[0]
    pr = model.predict(X_dev)
    print(np.array(pr).shape)
    # print(np.array(pr).shape)
    pr = np.argmax(pr, axis=-1)
    pred = np.zeros((pr.shape[0], pr.shape[1], height, width), dtype=int)
    for i in range(n_ops - 1):
        pred[:, :, :, :] = ((pr == i + 1) * 255).astype(int)
    # sio.savemat(dir_path+'pr_dev.mat', {'y_pred':pred,'name_dev':name})

    return pred


def Segnet(n_labels=2, img_h=68, img_w=68):
    kernel = 3
    img_in = Input(shape=(img_h, img_w, 3))
    
    # Load VGG16 pretrained on ImageNet data without including the top dense layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
    
    # Freeze VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Extract features from VGG16
    vgg_output = base_model(img_in)
    upsampled_output = UpSampling2D(size=(32,32))(vgg_output) 
    
    base_model.summary()
    
    # Add your custom layers after VGG16
    c1 = Conv2D(64, (kernel, kernel), padding='same')(upsampled_output)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    c2 = Conv2D(64, (kernel,kernel), padding='same')(a1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    m1 = MaxPooling2D(padding='same')(a2)
    # 12*34*34 
    c3 = Conv2D(128, (kernel, kernel), padding='same')(m1)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)
    c4 = Conv2D(128, (kernel, kernel), padding='same')(a3)
    b4 = BatchNormalization()(c4)
    a4 = Activation('relu')(b4)
    m2 = MaxPooling2D(padding='same')(a4)
    # 6*17*17
    c5 = Conv2D(256, (kernel, kernel), padding='same')(m2)
    b5 = BatchNormalization()(c5)
    a5 = Activation('relu')(b5)
    c6 = Conv2D(256, (kernel, kernel), padding='same')(a5)
    b6 = BatchNormalization()(c6)
    a6 = Activation('relu')(b6)
    c7 = Conv2D(256, (kernel, kernel), padding='same')(a6)
    b7 = BatchNormalization()(c7)
    a7 = Activation('relu')(b7)
    m3 = MaxPooling2D(padding='same')(a7)
    # 3*8*8
    
    ####decoder
    ###for mask1
    #6x17x17
    c8 = Conv2DTranspose( 512 , kernel_size=(kernel,kernel),strides=(2,2))(m3)
    c9 = Conv2D(512, (kernel, kernel), padding='same')(c8)
    b8 = BatchNormalization()(c9)
    a8 = Activation('relu')(b8)
    # 12*34*34	
    u1 = UpSampling2D()(a8)
    c12 = Conv2D(256, (kernel, kernel), padding='same')(u1)
    b11 = BatchNormalization()(c12)
    a11 = Activation('relu')(b11)
    # 24*68*68
    u2 = UpSampling2D()(a11) ## bs*24*68*68*
    c15 = Conv2D(128, (kernel, kernel), padding='same')(u2)
    b14 = BatchNormalization()(c15)
    a14 = Activation('relu')(b14)
    c18 = Conv2D(n_labels, (1, 1), padding='valid')(a14)
    b16 = BatchNormalization()(c18)
    s1 = Activation('softmax')(b16)	
    ###for mask2
        #6x17x17
    mc1 = Conv2DTranspose( 512 , kernel_size=(kernel,kernel) ,  strides=(2,2))(m3)
    mc2 = Conv2D(512, (kernel, kernel), padding='same')(mc1)
    mb1 = BatchNormalization()(mc2)
    ma1 = Activation('relu')(mb1)	
    # 12*34*34	
    mu1 = UpSampling2D()(ma1)
    mc3 = Conv2D(256, (kernel, kernel), padding='same')(mu1)
    mb2 = BatchNormalization()(mc3)
    ma2 = Activation('relu')(mb2)
    # 24*68*68
    mu2 = UpSampling2D()(ma2) ## bs*24*68*68*
    mc4 = Conv2D(128, (kernel, kernel), padding='same')(mu2)
    mb3 = BatchNormalization()(mc4)
    ma3 = Activation('relu')(mb3)
    mc5 = Conv2D(n_labels, (1, 1), padding='valid')(ma3)
    mb4 = BatchNormalization()(mc5)
    s2 = Activation('softmax')(mb4)	
    ###for mask3
        #6x17x17
    mc6 = Conv2DTranspose( 512 , kernel_size=(kernel,kernel) ,  strides=(2,2))(m3)
    mc7 = Conv2D(512, (kernel, kernel), padding='same')(mc6)
    mb5 = BatchNormalization()(mc7)
    ma5 = Activation('relu')(mb5)	
    # 12*34*34	
    mu3 = UpSampling2D()(ma5)
    mc8 = Conv2D(256, (kernel, kernel), padding='same')(mu3)
    mb6 = BatchNormalization()(mc8)
    ma6 = Activation('relu')(mb6)
    # 24*68*68
    mu4 = UpSampling2D()(ma6) ## bs*24*68*68*
    mc9 = Conv2D(128, (kernel, kernel), padding='same')(mu4)
    mb7 = BatchNormalization()(mc9)
    ma7 = Activation('relu')(mb7)
    mc10 = Conv2D(n_labels, (1, 1), padding='valid')(ma7)
    mb8 = BatchNormalization()(mc10)
    s3 = Activation('softmax')(mb8)	
    model = keras.models.Model(img_in,[s1,s2,s3])
    
    # Continue your Segnet architecture...

    #model = Model(img_in, [s1, s2, s3])
    return model


n_ops = 2
height=68
width=68
epoch =30
new = 1

video_paths = [
    f'74subjects_data/Videos Scaled to 68/FS1sub5.mp4',
    f'74subjects_data/Videos Scaled to 68/MS1sub6.mp4' 
    ]

annotation_paths = [
    f'74subjects_data/Matmasks_68/FS1sub5_mask_gt.mat',
    f'74subjects_data/Matmasks_68/MS1sub6_mask_gt.mat'
    ]

X, Y = ImageSegmentationGenerator(video_paths, annotation_paths, n_ops)

'''
# 91 frames is split into first 64 frames for training and last 27 frames for validation
# Split data for training (first 64 frames) and validation (remaining frames)
X_train = np.concatenate([X[:64], X[91:91+64]])  
X_val = np.concatenate([X[64:91], X[91+64:]])  

Y_train = [np.concatenate([Y[i][:64], Y[i][91:91+64]], axis=0) for i in range(3)]
Y_val = [np.concatenate([Y[i][64:91], Y[i][91+64:]], axis=0) for i in range(3)]
'''

# Ensure reproducibility
np.random.seed(1)  

# Number of frames in each set
frames_per_set = 91
train_frames = 64

# Randomly select indices for training and validation for the first set (0 to 90)
indices_first_set = np.random.choice(frames_per_set, train_frames, replace=False)
indices_first_val = np.setdiff1d(np.arange(frames_per_set), indices_first_set)

# Randomly select indices for training and validation for the second set (91 to 181)
indices_second_set = np.random.choice(frames_per_set, train_frames, replace=False)
indices_second_val = np.setdiff1d(np.arange(frames_per_set), indices_second_set)

# Splitting X based on the randomly selected indices
X_train = np.concatenate([X[indices_first_set], X[91 + indices_second_set]])
X_val = np.concatenate([X[indices_first_val], X[91 + indices_second_val]])

# Splitting Y based on the randomly selected indices for each of the three sets in Y
Y_train = [
    np.concatenate([Y[i][indices_first_set], Y[i][91 + indices_second_set]], axis=0)
    for i in range(3)
]
Y_val = [
    np.concatenate([Y[i][indices_first_val], Y[i][91 + indices_second_val]], axis=0)
    for i in range(3)
]



model_dir = f'/Pretrained_Models/Seen_Recording/1 videos/weights/'



print ('X_train.shape',X_train.shape)
print ('Y_train[0].shape',Y_train[0].shape)
print ('Y_train[1].shape',Y_train[1].shape)
print ('Y_train[2].shape',Y_train[2].shape)
print ('X_val.shape',X_val.shape)
print ('Y_val[0].shape',Y_val[0].shape)
print ('Y_val[1].shape',Y_val[1].shape)
print ('Y_val[2].shape',Y_val[2].shape)

print('*********new model fitting*********')
segnet_model=Segnet(n_ops ,height, width)
# segnet_model.summary()
segnet_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('compilation completed')
callbacks = [
    # ModelCheckpoint(model_dir+'model_best.weights', monitor='val_loss', verbose=0, save_best_only=True, mode = 'auto',period=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]		
print('callbacks completed')
history=segnet_model.fit(X_train, Y_train, epochs=epoch,batch_size=8,validation_data = (X_val, Y_val),verbose=1,callbacks=callbacks)
segnet_model.save(model_dir+'model_best.weights')
