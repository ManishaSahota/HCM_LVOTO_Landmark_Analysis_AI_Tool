import os
import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
import monai
import pandas as pd
from skimage.measure import centroid, label
from scipy.ndimage import gaussian_filter

# get the training, validation and test case list
def case_init(idx):
    case_file = 'case_list_5_fold.mat'
    # load case list
    case_list = sio.loadmat(case_file)
    trn_list, val_list, tst_list = case_list['f'][idx]
    return trn_list, val_list, tst_list

# get test data 
def load_data_test(idx):
    # idx = 0
    trn_list, val_list, tst_list = case_init(idx)
    # folder path
    data_folder_1 = "C:/Users/Manisha/Desktop/final_exp3/All Data/Ryan First 20/heatmaps"
    data_folder_2 = "C:/Users/Manisha/Desktop/final_exp3/All Data/Ryan Next 15/heatmaps"
    data_folder_3 = "C:/Users/Manisha/Desktop/final_exp3/All Data/Manisha First 20/heatmaps"
    data_folder_4 = "C:/Users/Manisha/Desktop/final_exp3/All Data/Manisha Next 15/heatmaps"
    # testing data
    tst_file_list = []
    for k_tst_idx in range(len(tst_list)):
        k_tst = 'HCMR_' + tst_list[k_tst_idx][0:3] + '_' + tst_list[k_tst_idx][4:]
        file_list = [os.path.join(data_folder_1, x) for x in os.listdir(data_folder_1) if k_tst in x
                     and 'heatmap_6_normal.mat' in x] + \
                    [os.path.join(data_folder_2, x) for x in os.listdir(data_folder_2) if k_tst in x
                     and 'heatmap_6_normal.mat' in x] + \
                    [os.path.join(data_folder_3, x) for x in os.listdir(data_folder_3) if k_tst in x
                     and 'heatmap_6_normal.mat' in x] + \
                    [os.path.join(data_folder_4, x) for x in os.listdir(data_folder_4) if k_tst in x
                     and 'heatmap_6_normal.mat' in x]
        for kf in file_list:
            tst_file_list.append(kf)
    test_x = np.zeros((len(tst_file_list) * 5, 1, 256, 256))
    test_y = np.zeros((len(tst_file_list) * 5, 14, 256, 256))
    c_tst = 0
    for file_1 in tst_file_list:
        # load file
        data_1 = sio.loadmat(file_1)
        x_1 = data_1['x']
        y_1 = data_1['y']
        for kn in range(5):
            test_x[c_tst:c_tst + 1, 0, :, :] = x_1[:, :, kn]
            test_y[c_tst:c_tst + 1, :, :, :] = np.transpose(y_1[:, :, kn, :], (2, 0, 1))
            c_tst += 1
    return tst_file_list, test_x, test_y

# extract centroids of the largest connected components from heatmaps 
def centroid_compare(prd, ref):
    centroid_prd = np.zeros((5, 14, 2))
    centroid_ref = np.zeros((5, 14, 2))
    # loop over all frames and landmarks
    for kf in range(5):
        for kp in range(14):
            prd_1 = prd[kf, kp, ...]
            # normalise the heatmaps to intensity range [0 255]
            prd_1 = prd_1 - np.min(prd_1)
            prd_1 = 255 * prd_1 / np.max(prd_1)
            # apply binary mask of threshold 128
            prd_1_ = np.zeros_like(prd_1)
            prd_1_[np.where(prd_1 >= 128)] = 1
            # get the largest connected component corresponding to the most likely landmark
            labels = label(prd_1_)
            l = np.unique(labels)
            n = [len(np.where(labels == x)[0]) for x in l]
            i = np.argmax(np.array(n[1:]))
            prd_1_1 = np.zeros_like(prd_1_)
            prd_1_1[np.where(labels == l[i + 1])] = 1
            # calculate centroid
            c_prd_1 = centroid(prd_1_)
            ref_1 = ref[kf, kp, ...]
            ref_1_ = np.zeros_like(ref_1)
            ref_1_[np.where(ref_1 >= 0.5)] = 1
            c_ref_1 = centroid(ref_1_)
            centroid_prd[kf, kp, :] = c_prd_1
            centroid_ref[kf, kp, :] = c_ref_1
    return centroid_prd, centroid_ref

# convolution
def k_conv():
    k = np.zeros((30, 30))
    k[14:16, 14:16] = 1
    lab_fp = np.zeros_like(k)
    lab_fp[np.where(k == 1)] = 1e6
    # convolution with Gaussian kernel filter of size sigma
    lab_fp = gaussian_filter(lab_fp, sigma=2)
    # normalise intensity
    k = lab_fp / np.max(lab_fp)
    return

# testing model
def test_models():
    device = torch.device("cpu")
    
    # get the best model from each fold of five fold cross validation
    test_model_list = ["C:/Users/Manisha/Desktop/final_exp3/model/f1/net_params_53_0.7625_22.6517.pth",
                       "C:/Users/Manisha/Desktop/final_exp3/model/f2/net_params_48_0.9922_31.0019.pth",
                       "C:/Users/Manisha/Desktop/final_exp3/model/f3/net_params_58_0.5601_39.5648.pth",
                       "C:/Users/Manisha/Desktop/final_exp3/model/f4/net_params_98_0.2721_35.3259.pth",
                       "C:/Users/Manisha/Desktop/final_exp3/model/f5/net_params_38_2.1036_38.0571.pth"]
    # empty arrays
    c_folder = []
    c_case = []
    c_frame = []
    c_landmark = []
    c_x_prd = []
    c_y_prd = []
    c_x_ref = []
    c_y_ref = []

    for k_idx in range(5):
        # define model
        test_model = test_model_list[k_idx]
        model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=1,
            out_channels=14,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2))
        model.load_state_dict(torch.load(test_model))
        model.to(device, dtype=torch.float)

        # load test data
        tst_file_list, test_x, test_y = load_data_test(k_idx)
        
        prd_y = np.zeros_like(test_y)
        # get predicted heatmaps from model
        for k_tst in range(len(tst_file_list)):
            t_x = Variable(torch.from_numpy(test_x[k_tst * 5:(k_tst + 1) * 5, ...]).float())
            output = model(t_x)
            prd = output.to('cpu').detach().numpy()
            prd_y[k_tst * 5:(k_tst + 1) * 5, ...] = prd
        # loop over test cases
        for k in range(len(tst_file_list)):
            print(tst_file_list[k])
            y_i = test_y[k * 5:(k + 1) * 5, ...]
            p_i = prd_y[k * 5:(k + 1) * 5, ...]
            # extract centroids of reference and predicted landmarks
            centroid_prd, centroid_ref = centroid_compare(p_i, y_i)
            # file name
            file_name = tst_file_list[k]
            # data path
            data_folder = "C:/Users/Manisha/Desktop/All Data"
            # folder name
            folder_name = file_name[file_name.find(data_folder) + len(data_folder):file_name.find('/heatmaps')]
            # case name
            case_name = file_name[file_name.find('/heatmaps') + 10:file_name.find('_heatmap_6_normal.mat')]
            # loop over all frames and landmarks 
            for ka in range(5):
                for kb in range(14):
                    # get the predicted and reference landmarks in x and y
                    x_prd = centroid_prd[ka, kb, 0]
                    y_prd = centroid_prd[ka, kb, 1]
                    x_ref = centroid_ref[ka, kb, 0]
                    y_ref = centroid_ref[ka, kb, 1]
                    # store variables
                    c_folder.append(folder_name)
                    c_case.append(case_name)
                    c_frame.append(ka + 1)
                    c_landmark.append(kb + 1)
                    c_x_prd.append(x_prd)
                    c_y_prd.append(y_prd)
                    c_x_ref.append(x_ref)
                    c_y_ref.append(y_ref)
    # print reference and predicted landmarks to csv file
    d = {'col1': c_folder, 'col2': c_case, 'col3': c_frame, 'col4': c_landmark, 'col5': c_x_prd, 'col6': c_y_prd,
         'col7': c_x_ref, 'col8': c_y_ref}
    df = pd.DataFrame(data=d)
    df.to_csv('test_result.csv')
    return
