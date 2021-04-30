import os
import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
import monai
import random
import nibabel as nib
from scipy.ndimage import gaussian_filter

# get the training, validation and test case list
def case_init(idx):
    case_file = 'case_list_5_fold.mat'
    # load case list
    case_list = sio.loadmat(case_file)
    trn_list, val_list, tst_list = case_list['f'][idx]
    return trn_list, val_list, tst_list

# data augmentation: intensity normalisation
def data_aug(img_i, ldm_i):
    # max percentile
    pmax = 95 + random.choice(list(np.array(list(range(10))) - 5))
    # rescale image intensity to max percentile
    img_fi = img_i / np.percentile(img_i, pmax)
    img_fi[np.where(img_fi > 1)] = 1
    x_i = img_fi
    ldm_fi = ldm_i
    y_i = np.zeros((np.shape(ldm_fi)[0], np.shape(ldm_fi)[1], np.max(ldm_fi).astype(int)))
    # extract landmarks
    for k in range(np.max(ldm_fi).astype(int)):
        pi = np.zeros_like(ldm_fi)
        pi[np.where(ldm_fi == k + 1)] = 1e6
        # convolution with Gaussian filter kernel of size sigma
        pi = gaussian_filter(pi, sigma=5)
        # normalise
        pi = pi / np.max(pi)
        y_i[..., k] = pi
    return x_i, y_i

# load annotated nii files
def load_data(file_1):
    img_file = file_1
    path_1, file_name = os.path.split(file_1)
    path_2, n = os.path.split(path_1)
    ldm_file = os.path.join(path_2, 'Annotated', file_name[0:file_name.find('.nii.gz')] + '_annotated.nii.gz')
    # load image nii file
    img_data = nib.load(img_file)
    # get image data
    img = img_data.get_data()
    # load annotated nii file
    ldm_data = nib.load(ldm_file)
    # get annotated landmark data
    ldm = ldm_data.get_data()
    # get annotated frame numbers 
    nf = np.unique(np.where(ldm)[2])
    x = np.zeros((128, 128, 5))
    y = np.zeros((128, 128, 5, 14))
    c = 0
    # loop over all landmarks in all frames
    for kf in nf:
        img_i = img[..., kf]
        ldm_i = ldm[..., kf]
        # data augementation
        x_i, y_i = data_aug(img_i, ldm_i)
        x[..., c] = x_i
        y[..., c, :] = y_i
        c += 1
    return x, y

# get the training and validating data
def load_data_train(idx):
    # idx = 0
    trn_list, val_list, tst_list = case_init(idx)
    # folder path
    data_folder_1 = 'E:\\Landmark_HCMR\\data\\crop_data\\Ryan First 20'
    data_folder_2 = 'E:\\Landmark_HCMR\\data\\crop_data\\Ryan Next 15'
    data_folder_3 = 'E:\\Landmark_HCMR\\data\\crop_data\\Manisha First 20'
    data_folder_4 = 'E:\\Landmark_HCMR\\data\\crop_data\\Manisha Next 15'
    # get training data
    train_x = np.zeros((len(trn_list) * 5, 1, 128, 128))
    train_y = np.zeros((len(trn_list) * 5, 14, 128, 128))
    c_trn = 0
    for k_trn_idx in range(len(trn_list)):
        k_trn = 'HCMR_' + trn_list[k_trn_idx][0:3] + '_' + trn_list[k_trn_idx][4:]
        file_list = [os.path.join(data_folder_1, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_1, 'Original'))
                     if k_trn in x] + \
                    [os.path.join(data_folder_2, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_2, 'Original'))
                     if k_trn in x] + \
                    [os.path.join(data_folder_3, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_3, 'Original'))
                     if k_trn in x] + \
                    [os.path.join(data_folder_4, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_4, 'Original'))
                     if k_trn in x]
        file_1 = random.choice(file_list)
        # data_1 = sio.loadmat(file_1)
        # x_1 = data_1['x']
        # y_1 = data_1['y']
        x_1, y_1 = load_data(file_1)
        for kn in range(5):
            train_x[c_trn, 0, :, :] = x_1[:, :, kn]
            train_y[c_trn, :, :, :] = np.transpose(y_1[:, :, kn, :], (2, 0, 1))
            c_trn += 1
    # get validating data
    valid_x = np.zeros((len(val_list) * 5, 1, 128, 128))
    valid_y = np.zeros((len(val_list) * 5, 14, 128, 128))
    c_val = 0
    for k_val_idx in range(len(val_list)):
        k_val = 'HCMR_' + val_list[k_val_idx][0:3] + '_' + val_list[k_val_idx][4:]
        file_list = [os.path.join(data_folder_1, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_1, 'Original'))
                     if k_val in x] + \
                    [os.path.join(data_folder_2, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_2, 'Original'))
                     if k_val in x] + \
                    [os.path.join(data_folder_3, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_3, 'Original'))
                     if k_val in x] + \
                    [os.path.join(data_folder_4, 'Original', x) for x in
                     os.listdir(os.path.join(data_folder_4, 'Original'))
                     if k_val in x]
        file_1 = random.choice(file_list)
        # data_1 = sio.loadmat(file_1)
        # x_1 = data_1['x']
        # y_1 = data_1['y']
        x_1, y_1 = load_data(file_1)
        for kn in range(5):
            valid_x[c_val, 0, :, :] = x_1[:, :, kn]
            valid_y[c_val, :, :, :] = np.transpose(y_1[:, :, kn, :], (2, 0, 1))
            c_val += 1
    return np.array(train_x), np.array(train_y), np.array(valid_x), np.array(valid_y)

# landmark detection training model
def train_model(epoch_init, epoch_end, idx, model_folder, lr=0.001, init_model=None):
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    training_loss = 0
    validating_loss = 0

    # define the model
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=14,
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
            txi = train_x[train_idx[k * 10:(k + 1) * 10], ...] * 255
            t_x = Variable(torch.from_numpy(txi).float().cuda())
            # t_x = Variable(torch.from_numpy(txi).float())
            tyi = train_y[train_idx[k * 10:(k + 1) * 10], ...] * 255
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

        vxi = valid_x * 255
        v_x = Variable(torch.from_numpy(vxi).float().cuda())
        # v_x = Variable(torch.from_numpy(vxi).float())
        vyi = valid_y * 255
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
model_folder = 'E:/Landmark_HCMR/model/crop/f' + str(idx + 1)
train_model(epoch_init, epoch_end, idx, model_folder)

epoch_init = 0
epoch_end = 100
idx = 1
model_folder = 'E:/Landmark_HCMR/model/crop/f' + str(idx + 1)
train_model(epoch_init, epoch_end, idx, model_folder)


epoch_init = 0
epoch_end = 100
idx = 2
model_folder = 'E:/Landmark_HCMR/model/crop/f' + str(idx + 1)
train_model(epoch_init, epoch_end, idx, model_folder)


epoch_init = 0
epoch_end = 100
idx = 3
model_folder = 'E:/Landmark_HCMR/model/crop/f' + str(idx + 1)
train_model(epoch_init, epoch_end, idx, model_folder)


epoch_init = 0
epoch_end = 100
idx = 4
model_folder = 'E:/Landmark_HCMR/model/crop/f' + str(idx + 1)
train_model(epoch_init, epoch_end, idx, model_folder)
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