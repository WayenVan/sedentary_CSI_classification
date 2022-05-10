import os
from socketserver import DatagramRequestHandler
import numpy as np
import scipy.io as scio




def normalize_data(data_1):
    # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
    data_1_max = np.concatenate((data_1.max(axis=0),data_1.max(axis=1)),axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0),data_1.min(axis=1)),axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max,(data_1.shape[0],data_1.shape[1],1))
    data_1_min_rep = np.tile(data_1_min,(data_1.shape[0],data_1.shape[1],1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return  data_1_norm

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())
    return np.array(data_pad)

def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     # from label to onehot
    return _label

def load_data(path_to_data, motion_sel):
    global T_MAX
    data = []
    label = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root,data_file_name)
            # try:
            data_1 = scio.loadmat(file_path)['save_spect']
            label_1 = int(data_file_name.split('-')[1])
                # location = int(data_file_name.split('-')[2])
                # orientation = int(data_file_name.split('-')[3])
                # repetition = int(data_file_name.split('-')[4])

                # Select Motion
            if (label_1 not in motion_sel):
                continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue
                
                # Normalization
            data_normed_1 = normalize_data(data_1)
                
                # Update T_MAX
            if T_MAX < np.array(data_1).shape[0]:
                T_MAX = np.array(data_1).shape[0]                
            #except Exception:
            #    continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1)
            
    # Zero-padding
    # data = zero_padding(data, T_MAX)

    # Swap axes
    # data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
    data = np.expand_dims(data, axis=-1)    # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1]

    # Convert label to ndarray
    label = np.array(label)

    # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
    return data, label

