import os
import cv2
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import scipy.io as sio
import matplotlib.pyplot as plt

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
        
# skip = True
# Define paths and parameters
video_paths = [
    f'74subjects_data/Videos Scaled to 68/FS1sub5.mp4',
    f'74subjects_data/Videos Scaled to 68/MS1sub6.mp4' 
    ]

annotation_paths = [
    f'74subjects_data/Matmasks_68/FS1sub5_mask_gt.mat',
    f'74subjects_data/Matmasks_68/MS1sub6_mask_gt.mat'
    ]



epochs = 30  # Maximum number of epochs for fine-tuning
batch_size = 4  # Set batch size for fine-tuning
n_ops = 2
height=68
width=68
new = 1


X, Y = ImageSegmentationGenerator(video_paths, annotation_paths, n_ops)

    
# 91 frames is split into first 45 frames for picking frames for fine-tuning and last 46 frames for validation

# Split data into training (first 45 frames) and validation (remaining frames)
X_new = np.concatenate([X[:45], X[91:91+45]])  # 1-45 frames has video1 and 46-90 frames has video2
X_val = np.concatenate([X[45:91], X[91 + 45:]])  # Remaining frames of each video

Y_new = [np.concatenate([Y[i][:45], Y[i][91:91+45]], axis=0) for i in range(3)]
Y_val = [np.concatenate([Y[i][45:91], Y[i][91 + 45:]], axis=0) for i in range(3)]


lists = [
    ['F1M1','F12M12','F123M123','F1234M1234']
]

base_models = lists[0]
print(base_models)

videos_for_training = ['2','4','8']
frames_for_tuning = [1,5,10,15]
runs = 10

ff, mf = 45, 45

for m in base_models:
    for v_num in videos_for_training:
        
        model_weights_path = f'Pretrained_Models/{m}/{v_num} videos/weights/model_best.weights' 
        
        for num_frames_to_select in frames_for_tuning:  # Number of random frames to select from each video
            for run in range(runs):
                
                # # Check if we should skip this iteration
                # if skip:
                #     # Check if the current condition matches the skip condition
                #     # print('in')
                #     if s_n == 4 and m == 'F12M12' and v_num == '2' and num_frames_to_select == 10 and run == 3:
                #         # print('out')
                #         skip = False  # Stop skipping after this condition is met
                #     continue
                
                # print('hi')
                main_model_save_dir = f'Finetuned_Weights/Unseen_Recording/{m}/{v_num}videos/{str(num_frames_to_select)}frames/run{run+1}'  

                # Ensure model save and plot directories exist
                if not os.path.exists(main_model_save_dir):
                    os.makedirs(main_model_save_dir)

                # Select M random frames from index range 0 to frames of F1
                indices1 = np.random.choice(range(0, ff), size=num_frames_to_select, replace=False)

                # Select M random frames from index range frames of F1 to M1
                indices2 = np.random.choice(range(ff, ff+mf), size=num_frames_to_select, replace=False)

                # Concatenate the selected indices
                selected_indices = np.concatenate((indices1, indices2))

                X_ft = X_new[selected_indices]

                Y_ft = [[],[],[]]
                Y_ft[0], Y_ft[1], Y_ft[2] = Y_new[0][selected_indices], Y_new[1][selected_indices], Y_new[2][selected_indices]

                # Load the pre-trained model and its weights
                model = load_model(model_weights_path)
                saveloc = main_model_save_dir
                print(saveloc)

                # Define callbacks for fine-tuning
                callbacks = [
                    # ModelCheckpoint(os.path.join(saveloc, 'model_finetuned.weights'), monitor='val_loss', save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)  # Early stopping with patience of 3 epochs
                ]

                # Fine-tune the model
                history = model.fit(
                    X_ft, 
                    Y_ft, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val),
                    verbose=1, 
                    callbacks=callbacks
                )

                # Save the fine-tuned model
                model.save(os.path.join(saveloc, 'model_finetuned.weights'))

                # Plot training and validation loss
                plt.figure()
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend(loc='upper right')

                # Save the loss curve plot
                loss_plot_save_path = os.path.join(saveloc, 'loss_curve.png')
                plt.savefig(loss_plot_save_path)
                print(f"Training and validation loss curve saved to {loss_plot_save_path}")

