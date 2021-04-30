import os
import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
import monai
import random
from img_aug_roi import data_aug_mat

# get the training, validation and test case list
def case_init(idx):
    case_file = 'case_list_5_fold.mat'
    # load case list 
    case_list = sio.loadmat(case_file)
    trn_list, val_list, tst_list = case_list['f'][idx]
    # load data folder
    data_folder = 'E:\\Landmark_HCMR\\data\\roi'
    case_list = os.listdir(data_folder)
    case_list = [x[0:x.find('.mat')] for x in case_list]
    trn_list_ = []
    val_list_ = []
    tst_list_ = []
    # get training, validation and test case list 
    for k_trn_idx in range(len(trn_list)):
        k_trn = 'HCMR_' + trn_list[k_trn_idx][0:3] + '_' + trn_list[k_trn_idx][4:]
        trn_list_.append(k_trn)
    for k_val_idx in range(len(val_list)):
        k_val = 'HCMR_' + val_list[k_val_idx][0:3] + '_' + val_list[k_val_idx][4:]
        val_list_.append(k_val)
    for k_tst_idx in range(len(tst_list)):
        k_tst = 'HCMR_' + tst_list[k_tst_idx][0:3] + '_' + tst_list[k_tst_idx][4:]
        tst_list_.append(k_tst)
    ex_list = [x for x in case_list if x not in trn_list_]
    ex_list = [x for x in ex_list if x not in val_list_]
    ex_list = [x for x in ex_list if x not in tst_list_]
    trn_list = trn_list_ + ex_list
    val_list = val_list_
    tst_list = tst_list_
    return trn_list, val_list, tst_list

# get the training and validation data
def load_data_train(idx):
    # idx = 0
    # data folder
    data_folder = 'E:\\Landmark_HCMR\\data\\roi'
    trn_list, val_list, tst_list = case_init(idx)
    # get training data
    train_x = np.zeros((len(trn_list) * 10, 1, 256, 256))
    train_y = np.zeros((len(trn_list) * 10, 4, 256, 256))
    c_trn = 0
    for k_trn_idx in range(len(trn_list)):
        file_1 = os.path.join(data_folder, trn_list[k_trn_idx] + '.mat')
        x, y = data_aug_mat(file_1)
        for kf in range(10):
            train_x[c_trn, 0, :, :] = x[:, :, kf]
            for kp in range(4):
                train_y[c_trn, kp, :, :] = y[:, :, kp, kf]
            c_trn += 1
    # validation data
    valid_x = np.zeros((len(val_list) * 10, 1, 256, 256))
    valid_y = np.zeros((len(val_list) * 10, 4, 256, 256))
    c_val = 0
    for k_val_idx in range(len(val_list)):
        file_1 = os.path.join(data_folder, val_list[k_val_idx] + '.mat')
        x, y = data_aug_mat(file_1)
        for kf in range(5):
            valid_x[c_val, 0, :, :] = x[:, :, kf]
            for kp in range(4):
                valid_y[c_val, kp, :, :] = y[:, :, kp, kf]
            c_val += 1
    return train_x, train_y, valid_x, valid_y

# ROI training model
def train_model(epoch_init, epoch_end, model_folder, idx, lr=0.001, init_model=None):
    device = torch.device('cuda:0')

    training_loss = 0
    validating_loss = 0

    # define the model 
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=4,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2))
    if init_model is not None:
        model.load_state_dict(torch.load(init_model))
    model.to(device, dtype=torch.float)
    model.cuda()
    # Adam optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE loss function
    loss_func = torch.nn.MSELoss()

    # training process
    for epoch in range(epoch_init, epoch_end, 1):
        print('Epoch: ' + str(epoch + 1))
        # load data
        train_x, train_y, valid_x, valid_y = load_data_train(idx)
        train_idx = random.sample(list(range(len(train_x))), len(train_x))
        # define the batch size
        batch_size = 10
        for k in range(int(len(train_idx) / batch_size)):
            txi = train_x[train_idx[k * 10:(k + 1) * 10], ...]
            t_x = Variable(torch.from_numpy(txi).float().cuda())
            # t_x = Variable(torch.from_numpy(txi).float())
            tyi = train_y[train_idx[k * 10:(k + 1) * 10], ...]
            t_y = Variable(torch.from_numpy(tyi).float().cuda())
            # t_y = Variable(torch.from_numpy(tyi).float())
            output = model(t_x)
            # compute the training loss
            loss = loss_func(output, t_y)
            # Adam optimiser
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # training loss
            training_loss += loss.item()
            print(k + 1)
            print(loss.item())
            del t_x, t_y, output, loss
            torch.cuda.empty_cache()

        vxi = valid_x
        v_x = Variable(torch.from_numpy(vxi).float().cuda())
        # v_x = Variable(torch.from_numpy(vxi).float())
        vyi = valid_y
        v_y = Variable(torch.from_numpy(vyi).float().cuda())
        # v_y = Variable(torch.from_numpy(vyi).float())
        output = model(v_x)
        # compute the validation loss
        loss = loss_func(output, v_y)
        # validation loss
        validating_loss += loss.item()
        print('val')
        print(loss.item())
        del v_x, v_y, output, loss
        torch.cuda.empty_cache()
        # print epoch number, training and validation loss  
        print('[%d] training_loss: %.4f, validating_loss: %.4f' %
              (epoch + 1, training_loss / (len(train_idx) / 10), validating_loss))
        # save model file
        save_file = os.path.join(model_folder, 'net_params_' + str(epoch + 1) + '_' +
                                 str(np.round(training_loss / (len(train_idx) / 10), 4)) + '_' +
                                 str(np.round(validating_loss, 4)) + '.pth')
        torch.save(model.state_dict(), save_file)
        training_loss = 0
        validating_loss = 0
    return


'''
epoch_init = 0
epoch_end = 100
idx = 0
model_folder = 'E:/Landmark_HCMR/model/roi4/f' + str(idx + 1)
train_model(epoch_init, epoch_end, model_folder, idx)

epoch_init = 0
epoch_end = 100
idx = 1
model_folder = 'E:/Landmark_HCMR/model/roi4/f' + str(idx + 1)
train_model(epoch_init, epoch_end, model_folder, idx)

epoch_init = 0
epoch_end = 100
idx = 2
model_folder = 'E:/Landmark_HCMR/model/roi4/f' + str(idx + 1)
train_model(epoch_init, epoch_end, model_folder, idx)

epoch_init = 0
epoch_end = 100
idx = 3
model_folder = 'E:/Landmark_HCMR/model/roi4/f' + str(idx + 1)
train_model(epoch_init, epoch_end, model_folder, idx)

epoch_init = 0
epoch_end = 100
idx = 4
model_folder = 'E:/Landmark_HCMR/model/roi4/f' + str(idx + 1)
train_model(epoch_init, epoch_end, model_folder, idx)
'''


# loss plots
def loss_plot(model_folder):
    file_list = os.listdir(model_folder)
    epoch = []
    trn = []
    val = []
    list_sort = []
    # get the validation and training losses
    for k in range(len(file_list)):
        file_i = [x for x in file_list if 'net_params_' + str(k + 1) + '_' in x]
        file_1 = file_i[0]
        file_i = file_i[0][len('net_params_' + str(k + 1) + '_'):]
        trn_i = file_i[0:file_i.find('_')]
        file_i = file_i[len(trn_i) + 1:]
        val_i = file_i[0:file_i.find('.pth')]
        epoch.append(np.int(k + 1))
        trn.append(np.float(trn_i))
        val.append(np.float(val_i))
        list_sort.append(os.path.join(model_folder, file_1))
    # find the minimum validation loss and corresponding training loss
    model_idx = np.argmin(val)
    val_model = val[model_idx]
    trn_model = trn[model_idx]
    
    # plot training and validation loss vs number of epochs
    '''
    plt.figure()
    plt.plot(epoch, trn, label='Train')
    plt.plot(epoch, val, label='Valid')
    plt.scatter([model_idx + 1], [val_model], label='Min Val Loss')
    plt.scatter([model_idx + 1], [trn_model], label='Cor Trn Loss')
    plt.legend()
    '''
    # get the corresponding model file path
    model_path = list_sort[model_idx]
    return model_path, val_model, trn_model
