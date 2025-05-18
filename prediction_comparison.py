# Prediction of the benchmark models

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import cv2
import glob
import numpy as np
from keras.models import *
from keras.layers import *
import scipy.io as sio
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def predicted_output(model,X_dev, height=68,width=68,n_ops=2):
        batch_len=X_dev.shape[0]
        # print("X dev shape",  X_dev.shape)
        pr = model.predict(X_dev)
        # print(np.array(pr).shape)
        # print(np.array(pr).shape)
        pr = np.argmax(pr,axis=-1)
        pred = np.zeros((pr.shape[0],pr.shape[1],height,width),dtype = int)
        for i in range(n_ops-1):
                pred[:,:,:,:] = ((pr==i+1)*255).astype(int)
        #sio.savemat(dir_path+'pr_dev.mat', {'y_pred':pred,'name_dev':name})
        # print("dffdf", pred.shape)
        return pred


def compute_iou(truth, prediction):
    """
    Compute the Intersection over Union (IoU) between two binary images.
    
    Parameters:
    truth (numpy array): The first binary image.
    prediction (numpy array): The second binary image.
    
    Returns:
    float: The IoU score.
    """
    
    # Calculate intersection and union
    intersection = np.logical_and(truth, prediction).sum()
    union = np.logical_or(truth, prediction).sum()
    
    if union == 0:
        return 0.0
    
    # Compute IoU
    iou = intersection / union
    return iou


def compute_dice(truth, prediction):
    
    intersection = np.logical_and(truth, prediction).sum()
    area = truth.sum() + prediction.sum()
    
    if area == 0:
        return 0.0
    
    # Compute Dice
    dice = 2 * intersection / area
    return dice


def plot_scores(subjects, pca, iou, dice, contour_number):
        plt.scatter(subjects, pca, label='PCA')
        plt.scatter(subjects, iou, label='IoU')
        plt.scatter(subjects, dice, label='Dice')
        
        # ymin = np.min([min(pca), min(iou), min(dice)])
        
        plt.ylim([0, 1])
        plt.legend()
        plt.xlabel('Video')
        plt.ylabel('Scores')
        plt.title(f'Scores Comparison of Contour {contour_number}')
        plt.savefig(f'{pred_folder}/contour{contour_number}_scores.png')
        plt.show()

snums = [5]
# snums = [5,4,3,2,1]
for s_n in snums:
                
        subs = [f'F{s_n}',f'M{s_n}']
        
        base_models = [f'F{s_n}M{s_n}']

    	videos_for_training = ['2','4','8']
        

        height = 68
        width = 68
        n_ops = 2


        for m in base_models:
            for v_num in videos_for_training:
                # Pretrained_Models/F5M5/8 videos/weights

                        save_path=(f'<path to model weights for inference>')
                        mat_save = save_path + 'predicted_masks/'
                        print(save_path)
                        segnet_model = load_model(save_path + 'weights/model_best.weights')
                        # segnet_model.summary()

                        if not os.path.exists(mat_save):
                                                print("creating test directory : "+mat_save)
                                                os.makedirs(mat_save)

                        for sub in range(len(subs)):
                                videos = glob.glob(f'Unseen_Data/{subs[sub]}videos'+'/*.avi')
                                videos.sort()
                                vid = videos[2:]
                                for vl in range(len(vid)):
                                        # print("hello")
                                        vidcap = cv2.VideoCapture(vid[vl])
                                        a=[]
                                        names_test = []
                                        success = True
                                        while success:
                                                success,image = vidcap.read()
                                                a.append(image)
                                        a = np.array(a)
                                        b = np.zeros((a.shape[0]-1,68,68,3))
                                        for i in range(a.shape[0]-1):
                                                names_test.append(subs[sub]+'_'+vid[vl]+str(i).zfill(3))
                                                b[i,:,:,:] = a[i]
                                        b=b.astype('float64')
                                        b=b/255
                                        # if not os.path.exists(mat_save):
                                        #         print("creating test directory : "+mat_save)
                                        #         os.makedirs(mat_save)
                                        # else:
                                        #         print("image directory already exists")
                                        #b_test_rot=np.array(b_test_rot)
                                        #names_test_rot = np.array(names_test_rot)
                                        b=np.array(b)
                                        names_test = names_test[0:len(names_test)]
                                        names_test = np.array(names_test)
                                        #print(np.array(b).shape)
                                        #print(np.array(names_test).shape)
                                        preds = predicted_output(segnet_model,b,height,width,n_ops=2)
                                        # print(preds.shape)
                                        """for i in range(preds.shape[0]):
                                        for j in range(preds.shape[1]):
                                                for k in range(preds.shape[2]):
                                                for l in range(preds.shape[3]):
                                                        if preds[i][j][k][l] != 0:
                                                        print("haha")"""
                                        sio.savemat(mat_save+''+subs[sub]+'_'+vid[vl][len(vid[vl])-7:len(vid[vl])-4]+'.mat', {'pred':preds,'name':names_test})
                


                        pred_folder = mat_save
                        gt_folder = '/Unseen_Data/'  # Folder should contain the mat files directly without any subfolders
                        gt_subfolder = [f'F{s_n}masks',f'M{s_n}masks']
                        mat_files = [f'F{s_n}_003',f'F{s_n}_004',f'F{s_n}_005',f'M{s_n}_003',f'M{s_n}_004',f'M{s_n}_005']

                        clean_pca, clean_iou, clean_dice = np.zeros((3,6)), np.zeros((3,6)), np.zeros((3,6))
                        
                        for i in range(len(mat_files)):
                                
                                mat1 = sio.loadmat(pred_folder + f'{mat_files[i]}.mat')
                                
                                mat2 = sio.loadmat(gt_folder + gt_subfolder[i//3] + f'/{mat_files[i]}_mask_gt.mat')
                                # print(mat2.keys())
                                # print(mat2['masks'].shape)
                                
                                y = mat1['pred']  # contour no., frame no., height, width
                                x = mat2['masks_struct'][0]  # frame no., contour no.
                                
                                y = (y/y.max()).astype(int)
                                frames = x.shape[0]
                                        
                                frame_accuracy = np.zeros((9,frames), dtype=float)
                                
                                for contour in range(3):
                                        for frame in range(frames):
                                                frame_accuracy[contour, frame] = np.sum(x[frame][contour] == y[contour][frame]) / (68*68)
                                                frame_accuracy[contour+3,frame] = compute_iou(x[frame][contour], y[contour][frame])
                                                frame_accuracy[contour+6,frame] = compute_dice(x[frame][contour], y[contour][frame])
                                        
                                        clean_pca[contour][i] = np.mean(frame_accuracy[contour])
                                        clean_iou[contour][i] = np.mean(frame_accuracy[contour+3])
                                        clean_dice[contour][i] = np.mean(frame_accuracy[contour+6])


                        

                        # Plotting the scores for each contour
                        for i in range(3):
                                plot_scores(mat_files, clean_pca[i], clean_iou[i], clean_dice[i], i + 1)
                        
                        print('PCA: \ncontour1 ->', clean_pca[0].mean(),
                                '\ncontour2 ->', clean_pca[1].mean(),
                                '\ncontour3 ->', clean_pca[2].mean(),
                        '\nIoU: \ncontour1 ->', clean_iou[0].mean(),
                                '\ncontour2 ->', clean_iou[1].mean(),
                                '\ncontour3 ->', clean_iou[2].mean(),
                        '\nDice: \ncontour1 ->', clean_dice[0].mean(),
                                '\ncontour2 ->', clean_dice[1].mean(),
                                '\ncontour3 ->', clean_dice[2].mean())

                        # Define the file name and combine it with the folder path
                        file_path = os.path.join(pred_folder, 'countours_scores.txt')

                        # Open the file in write mode and create it in the specified folder
                        with open(file_path, 'w') as file:
                        # Prepare the content to be written
                                content = (f'PCA: \ncontour1 -> {clean_pca[0].mean()}'
                                        f'\ncontour2 -> {clean_pca[1].mean()}'
                                        f'\ncontour3 -> {clean_pca[2].mean()}'
                                        f'\nIoU: \ncontour1 -> {clean_iou[0].mean()}'
                                        f'\ncontour2 -> {clean_iou[1].mean()}'
                                        f'\ncontour3 -> {clean_iou[2].mean()}'
                                        f'\nDice: \ncontour1 -> {clean_dice[0].mean()}'
                                        f'\ncontour2 -> {clean_dice[1].mean()}'
                                        f'\ncontour3 -> {clean_dice[2].mean()}')
                                
                                # Write the content to the file
                                file.write(content)
