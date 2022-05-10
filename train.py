from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ViT import ViT
import torch
from sklearn.model_selection import train_test_split
from common import load_data,onehot_encoding,zero_padding,normalize_data
import time

# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = ''
ALL_MOTION = [1,2,3,4,5,6,7,8]
N_MOTION = len(ALL_MOTION)
T_MAX = 0
n_epochs = 50
f_dropout_ratio = 0.5
n_batch_size = 32
f_learning_rate = 0.001

# Load data
# data, label = load_data(data_dir, ALL_MOTION)


print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0,:,:].shape) + '\n')



# Split train and test
[data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +\
    'Test on ' + str(label_test.shape[0]) + ' samples\n')
data_train = DataLoader(data_train, batch_size = n_batch_size)
data_test = DataLoader(data_test)

# One-hot encoding for train data
label_train = onehot_encoding(label_train, N_MOTION)

# Load or fabricate model
if use_existing_model:
    model = torch.load('model_widar3_trained.h5')
    print(model)
else:
    model = ViT(input_shape=(T_MAX, 90, 90, 1), n_class=N_MOTION).cuda()
    print(model)
    model.train()
    for epoch in range(n_epochs)：
        time_start = time.time()

        for i, batch in enumerate(data_train, 0):

            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            outputs = model(X_train)


    
    x = model.forward
    model.fit({'name_model_input': data_train},{'name_model_output': label_train},
            batch_size=n_batch_size,
            epochs=n_epochs,
            verbose=1,
            validation_split=0.1, shuffle=True)
    print('Saving trained model...')
    model.save('model_widar3_trained.h5')

# x = torch.rand((10, 1500, 30, 30, 3), dtype=torch.float).cuda()
# model = ViT(2048, 2048, (30, 30, 3), 8, 12).cuda()
# x = model.forward(x)
# print(x)