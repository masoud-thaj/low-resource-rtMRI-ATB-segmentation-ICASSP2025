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


def imageSegmentationGenerator(in_dir, out_dir, n_ops, height, width, train_matrix, val_matrix, subs):
    assert in_dir[-1] == '/'
    assert out_dir[-1][-1] == '/'
    X_train = []
    Y_train = [[], [], []]
    X_val = []
    Y_val = [[], [], []]
    names = []
    names_val = []
    # To generate train data
    for sub in subs:
        
        for vid in train_matrix:
            print(vid)
            X_train, Y_train, names = generate_data(in_dir, out_dir, sub, vid, X_train, Y_train, names, height, width, n_ops)
            
        for vid in val_matrix:
            X_val, Y_val, names_val = generate_data(in_dir, out_dir, sub, vid, X_val, Y_val, names_val, height, width, n_ops)
    # print(len(X_train))
    # print(np.array(X_train).shape)
    X_train = np.array(X_train).astype('float64')
    X_train = X_train / 255
    X_val = np.array(X_val).astype('float64')
    X_val = X_val / 255
    Y_train1 = [np.array(Y_train[i]) for i in range(3)]
    Y_val1 = [np.array(Y_val[i]) for i in range(3)]
    return np.array(X_train), Y_train1, np.array(X_val), Y_val1, np.array(names), np.array(names_val)

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
    
    # Add custom layers after VGG16
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
    
    return model


 roundrobin_subjects = [
      'F1M1','F12M12','F123M123','F1234M1234',
        ]

 subs_list = [
      ['F1', 'M1'],
      ['F1', 'F2', 'M1', 'M2'],
      ['F1', 'F2', 'F3', 'M1', 'M2', 'M3'],
      ['F1', 'F2', 'F3', 'F4', 'M1', 'M2', 'M3', 'M4'],
     ]

videos_for_training = [2,4,6,8]


sub_count = -1

for subs in subs_list:
    sub_count += 1
    
    for tr_videos in videos_for_training:

        model_dir = f'Pretrained_Models/{roundrobin_subjects[sub_count]}/{tr_videos} videos/weights/'
        in_dir = ' ## videos directory'
        out_dir = ' ## masks matfiles directory'
        n_ops = 2
        height=68
        width=68
        epoch =30
        new = 1
        #segnet_model=model.Segnet(n_ops ,height, width)
        #for layer in segnet_model.layers:
            #print(layer.output_shape)
        if not os.path.exists(model_dir):
            print("creating model directory ")
            os.makedirs(model_dir)
            

        actual_set=np.array([342,391,392,393,394,395,397,398,399,406])

        train_matrix = actual_set[:tr_videos]
        if tr_videos < 5:
            val_matrix = actual_set[-2:-1]
        else:
            val_matrix = actual_set[-2:]

        X_train,Y_train,X_val,Y_val,name,name_val=imageSegmentationGenerator(in_dir,out_dir,n_ops,height,width,train_matrix,val_matrix,subs)
        # print ('X_train.shape',X_train.shape)
        # print ('Y_train[0].shape',Y_train[0].shape)
        # print ('Y_train[1].shape',Y_train[1].shape)
        # print ('Y_train[2].shape',Y_train[2].shape)
        # print ('X_val.shape',X_val.shape)
        # print ('Y_val[0].shape',Y_val[0].shape)
        # print ('Y_val[1].shape',Y_val[1].shape)
        # print ('Y_val[2].shape',Y_val[2].shape)
        # print ('name.shape', name.shape)
        # print ('name_val.shape', name_val.shape)
            
        print('*********new model fitting*********')
        segnet_model=Segnet(n_ops ,height, width)
        # segnet_model.summary()
        segnet_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        print('compilation completed')
        callbacks = [
            # ModelCheckpoint(model_dir+'model_best.weights', monitor='val_loss', verbose=0, save_best_only=True, mode = 'auto',period=1),
            EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            ]		
        print('callbacks completed')
        history=segnet_model.fit(X_train, Y_train, epochs=epoch,batch_size=8,validation_data = (X_val, Y_val),verbose=1,callbacks=callbacks)
        segnet_model.save(model_dir+'model_best.weights')
