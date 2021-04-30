import os
import numpy as np
import monai
import torch
from torch.autograd import Variable
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.measure import centroid, label
from PIL import Image
import nibabel as nib

# reorientate and flip image
def roi_case(x_o, p0_cm, p1_cm, p2_cm, p3_cm):
    img_0 = x_o[..., 0] * 255
    
    # convolution of each landmark with Gaussian filter kernel of size sigma
    # intensity rescaling 
    p0 = np.zeros_like(img_0)
    p0[int(p0_cm[0]), int(p0_cm[1])] = 1e6
    p0 = gaussian_filter(p0, sigma=10)
    p0 = p0 / np.max(p0) * 255
    p1 = np.zeros_like(img_0)
    p1[int(p1_cm[0]), int(p1_cm[1])] = 1e6
    p1 = gaussian_filter(p1, sigma=10)
    p1 = p1 / np.max(p1) * 255
    p2 = np.zeros_like(img_0)
    p2[int(p2_cm[0]), int(p2_cm[1])] = 1e6
    p2 = gaussian_filter(p2, sigma=10)
    p2 = p2 / np.max(p2) * 255
    p3 = np.zeros_like(img_0)
    p3[int(p3_cm[0]), int(p3_cm[1])] = 1e6
    p3 = gaussian_filter(p3, sigma=10)
    p3 = p3 / np.max(p3) * 255

    # calculate the centre of mass of the landmarks
    p_cm = np.mean([p0_cm, p1_cm, p2_cm, p3_cm], axis=0)
    # compute LV length and normalise
    v = p1_cm - p0_cm
    v_ = v / np.linalg.norm(v)
    
    # rotate image so that the line connecting the AML hinge and apex landmarks aligns to the vertical axis
    r = np.rad2deg(np.arctan(v_[0] / v_[1])) - 90
    img_i = Image.fromarray(img_0)
    img_r = img_i.rotate(r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
    p0_i = Image.fromarray(p0)
    p0_r = p0_i.rotate(r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
    p1_i = Image.fromarray(p1)
    p1_r = p1_i.rotate(r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
    p2_i = Image.fromarray(p2)
    p2_r = p2_i.rotate(r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
    p3_i = Image.fromarray(p3)
    p3_r = p3_i.rotate(r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
    # compute centroids for landmarks after rotation
    p0_rc = centroid(np.array(p0_r))
    p1_rc = centroid(np.array(p1_r))
    # check up-down direction of apex and AML hinge and flip if needed
    n = p1_rc - p0_rc
    if np.round(np.abs(n[0]) / np.linalg.norm(n)) == 1:
        if n[0] > 0:
            img_rf = np.array(img_r)
            p0_rf = np.array(p0_r)
            p1_rf = np.array(p1_r)
            p2_rf = np.array(p2_r)
            p3_rf = np.array(p3_r)
            f_ud = 0
        else:
            img_rf = np.flipud(np.array(img_r))
            p0_rf = np.flipud(np.array(p0_r))
            p1_rf = np.flipud(np.array(p1_r))
            p2_rf = np.flipud(np.array(p2_r))
            p3_rf = np.flipud(np.array(p3_r))
            f_ud = 1
    else:
        print('Check rotation.')
        
    # compute centroids for PML hinge and LVOT landmarks after rotation and up-down flip
    p2_rfc = centroid(np.array(p2_rf))
    p3_rfc = centroid(np.array(p3_rf))
    h = p2_rfc - p3_rfc
    # check left-right direction and flip if needed
    if h[1] > 0:
        img_rff = np.array(img_rf)
        p0_rff = np.array(p0_rf)
        p1_rff = np.array(p1_rf)
        p2_rff = np.array(p2_rf)
        p3_rff = np.array(p3_rf)
        f_lr = 0
    else:
        img_rff = np.fliplr(np.array(img_rf))
        p0_rff = np.fliplr(np.array(p0_rf))
        p1_rff = np.fliplr(np.array(p1_rf))
        p2_rff = np.fliplr(np.array(p2_rf))
        p3_rff = np.fliplr(np.array(p3_rf))
        f_lr = 1

    p_rff = np.zeros_like(img_rff)
    p_rff[p0_rff > 128] = 50
    p_rff[p1_rff > 128] = 100
    p_rff[p2_rff > 128] = 150
    p_rff[p3_rff > 128] = 200
    p4_rff = np.stack((p0_rff, p1_rff, p2_rff, p3_rff), axis=2)
    return p_cm, r, f_ud, f_lr, img_rff, p_rff, p4_rff

# get ROI
def img_roi(img, model_list):
    test_x = np.transpose(img, (2, 0, 1))[:, np.newaxis, ...]
    prd_y = np.zeros((np.shape(test_x)[0], 4, np.shape(test_x)[2], np.shape(test_x)[3]))
    # loop over models
    for model in model_list:
        prd_yi = np.zeros((np.shape(test_x)[0], 4, np.shape(test_x)[2], np.shape(test_x)[3]))
        # define batch size 
        batch_size = 1
        # get ROI predicted heatmaps from model
        for kf in range(int(len(test_x) / batch_size)):
            txi = test_x[kf * batch_size:(kf + 1) * batch_size, ...]
            t_x = Variable(torch.from_numpy(txi).float().cuda())
            output = model(t_x)
            prd = output.to('cpu').detach().numpy()
            prd_yi[kf * batch_size:(kf + 1) * batch_size, ...] = prd
            del t_x, output
            torch.cuda.empty_cache()
        # aggregate predictions and take the average as the final prediction     
        prd_y += prd_yi
    prd_y = prd_y / len(model_list)
    prd_y_ = np.zeros_like(prd_y)
    for kf in range(np.shape(prd_y)[0]):
        for kl in range(np.shape(prd_y)[1]):
            prd_yi = prd_y[kf, kl, ...]
            # normalise intensity
            prd_yi0 = prd_yi - np.min(prd_yi)
            prd_yi1 = prd_yi0 / np.max(prd_yi0)
            prd_y_[kf, kl, ...] = prd_yi1
    prd_y = 255 * prd_y_

    p0 = []
    p1 = []
    p2 = []
    p3 = []
    # apply intensity threshold of 128 to predicted heatmaps 
    # extract centroid of largest connected component
    for kf in range(len(prd_y)):
        p0_i = prd_y[kf, 0, ...]
        p0_c = centroid(p0_i > 128)
        p1_i = prd_y[kf, 1, ...]
        p1_c = centroid(p1_i > 128)
        p2_i = prd_y[kf, 2, ...]
        p2_c = centroid(p2_i > 128)
        p3_i = prd_y[kf, 3, ...]
        p3_c = centroid(p3_i > 128)
        if not (np.isnan(p0_c[0]) or np.isnan(p0_c[1])):
            p0.append(p0_c)
        if not (np.isnan(p1_c[0]) or np.isnan(p1_c[1])):
            p1.append(p1_c)
        if not (np.isnan(p2_c[0]) or np.isnan(p2_c[1])):
            p2.append(p2_c)
        if not (np.isnan(p3_c[0]) or np.isnan(p3_c[1])):
            p3.append(p3_c)
    # compute the median coordinates 
    p0_cm = np.median(p0, axis=0)
    p1_cm = np.median(p1, axis=0)
    p2_cm = np.median(p2, axis=0)
    p3_cm = np.median(p3, axis=0)
    return p0_cm, p1_cm, p2_cm, p3_cm

# transform image
def trans_forward(img, p_cm, r, f_ud, f_lr):
    img_rff_all = np.zeros_like(img)
    for kf in range(np.shape(img)[2]):
        img_i = Image.fromarray(img[..., kf])
        # rotate image
        img_r = img_i.rotate(r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
        # check up-down direction and flip if needed
        if f_ud == 1:
            img_rf = np.flipud(np.array(img_r))
        else:
            img_rf = np.array(img_r)
        # check lef-right direction and flip if needed 
        if f_lr == 1:
            img_rff = np.fliplr(np.array(img_rf))
        else:
            img_rff = np.array(img_rf)
        img_rff_all[..., kf] = img_rff
    return img_rff_all

# crop and resize to 128 x 128 
def img_roi_128(img_rff_all, p4_rff):
    # get the bounding box coordinates
    x_min, x_max, y_min, y_max = roi_xy(p4_rff)
    # crop image to bounding box size
    img_crop = img_rff_all[x_min:x_max, y_min:y_max, :]
    # resize cropped image to 128x128
    img_128 = np.zeros((128, 128, np.shape(img_rff_all)[2]))
    for kf in range(np.shape(img_rff_all)[2]):
        img_i = Image.fromarray(img_crop[..., kf])
        # resampling
        img_ir = np.array(img_i.resize((128, 128), resample=Image.NEAREST))
        # normalise
        img_ir0 = img_ir - np.min(img_ir)
        img_ir1 = img_ir0 / np.max(img_ir0)
        img_128[..., kf] = img_ir1
    return img_128

# get coordinates for the ROI bounding box
def roi_xy(p4_rff):
    # compute centroid to extract landmark coordinates for the AML hinge and apex
    p0_c = centroid(p4_rff[..., 0] > 128)
    p1_c = centroid(p4_rff[..., 1] > 128)
    # get point which lies at the midpoint of the two landmarks
    c = (p0_c + p1_c) / 2
    # calculate vector from landmark to centre
    vec = p1_c - c
    # calculate length from landmark to centre
    dist = np.sqrt((p1_c[1] - c[1]) ** 2 + (p1_c[0] - c[0]) ** 2)
    # add 50% to length to enclude entire heart
    lngth = 1.5 * dist
    # normalise the vector
    normVec = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    # find orthogonal vector
    orthVec1 = np.array([normVec[1], -normVec[0]])
    # get corner coordinates of the bounding box
    c1 = c + lngth * normVec + lngth * orthVec1
    c2 = c - lngth * normVec + lngth * orthVec1
    c3 = c + lngth * normVec - lngth * orthVec1
    c4 = c - lngth * normVec - lngth * orthVec1
    x_min = np.min([c1[0], c2[0], c3[0], c4[0]])
    x_max = np.max([c1[0], c2[0], c3[0], c4[0]])
    y_min = np.min([c1[1], c2[1], c3[1], c4[1]])
    y_max = np.max([c1[1], c2[1], c3[1], c4[1]])
    return x_min.astype(int), x_max.astype(int), y_min.astype(int), y_max.astype(int)

# generate the predicted heatmap
def prd_gen(img_128, model_ldm_list):
    prd_128 = np.zeros((128, 128, np.shape(img_128)[2], 14))
    # loop over all models
    for model_ldm in model_ldm_list:
        prd_128_i = np.zeros((128, 128, np.shape(img_128)[2], 14))
        # generate predicted heatmaps for each landmark from model
        for kn in range(np.shape(img_128)[2]):
            test_x = np.zeros((1, 1, 128, 128))
            test_x[0, 0, :, :] = img_128[:, :, kn]
            t_x = Variable(torch.from_numpy(test_x * 255).float().cuda())
            output = model_ldm(t_x)
            prd = output.to('cpu').detach().numpy()
            for kp in range(14):
                prd_128_i[:, :, kn, kp] = prd[0, kp, ...]
        # aggregate predictions and take the average as the final prediction 
        prd_128 += prd_128_i
    prd_128 = prd_128 / len(model_ldm_list)
    return prd_128

# transform cropped image back to original size 
def prd_roi_128_back(img_rff_all, p4_rff, prd_128):
    prd_rff_all_back = np.zeros((np.shape(img_rff_all)[0], np.shape(img_rff_all)[1], np.shape(img_rff_all)[2], 14))
    # get coordinates for the ROI bounding box
    x_min, x_max, y_min, y_max = roi_xy(p4_rff)
    # cropped image
    ldm_crop = img_rff_all[x_min:x_max, y_min:y_max, :]
    for kf in range(np.shape(img_rff_all)[2]):
        ldm_i_ = np.zeros((np.shape(ldm_crop)[0], np.shape(ldm_crop)[1], 14))
        # loop over all landmarks 
        for kn in range(14):
            # resize cropped image to match original size 
            ldm_ip = prd_128[..., kf, kn]
            ldm_i = Image.fromarray(ldm_ip)
            ldm_ir = np.array(ldm_i.resize((np.shape(ldm_crop)[1], np.shape(ldm_crop)[0]),
                                           resample=Image.NEAREST))
            ldm_i_[..., kn] = ldm_ir
        prd_rff_all_back[x_min:x_max, y_min:y_max, kf] = ldm_i_
    return prd_rff_all_back

# transform image back to original orientation
def trans_backward_ldm(prd_rff_all, p_cm, r, f_ud, f_lr):
    prd_all = np.zeros_like(prd_rff_all)
    nf = np.shape(prd_rff_all)[2]
    nl = np.shape(prd_rff_all)[3]
    for kf in range(nf):
        for kl in range(nl):
            ldm_rff = prd_rff_all[..., kf, kl]
            # check left-right direction and flip if needed to match original image
            if f_lr == 1:
                ldm_rf = np.fliplr(np.array(ldm_rff))
            else:
                ldm_rf = np.array(ldm_rff)
                # check up-down direction and flip if needed to match original image
            if f_ud == 1:
                ldm_r = np.flipud(np.array(ldm_rf))
            else:
                ldm_r = np.array(ldm_rf)
            ldm_i = Image.fromarray(ldm_r)
            # resampling
            ldm_r_ = ldm_i.rotate(-r, resample=Image.NEAREST, center=tuple([p_cm[1], p_cm[0]]))
            prd_all[..., kf, kl] = ldm_r_
    return prd_all

# get the predicted image
def img_prd_tst(img_file, model_roi_list, model_ldm_list):
    # load image nii file 
    img_data = nib.load(img_file)
    # get image data
    img = img_data.get_data()
    # get the ROI
    p0_cm, p1_cm, p2_cm, p3_cm = img_roi(img, model_roi_list)
    # reorientate and flip image
    p_cm, r, f_ud, f_lr, img_rff, p_rff, p4_rff = roi_case(img, p0_cm, p1_cm, p2_cm, p3_cm)
    # transform image
    img_rff_all = trans_forward(img, p_cm, r, f_ud, f_lr)
    # crop and resize to 128x128
    img_128 = img_roi_128(img_rff_all, p4_rff)
    # generate predicted heatmap
    prd_128 = prd_gen(img_128, model_ldm_list)
    # transform image back to original size
    prd_rff_all_back = prd_roi_128_back(img_rff_all, p4_rff, prd_128)
    # transform image back to original orientation 
    prd_all_back = trans_backward_ldm(prd_rff_all_back, p_cm, r, f_ud, f_lr)
    return img, prd_all_back

# get the best model filepath
def model_select(model_folder):
    # get file path
    file_list = os.listdir(model_folder)
    epoch = []
    trn = []
    val = []
    list_sort = []
    # loop over all model filepaths
    for k in range(len(file_list)):
        # get number of epochs
        file_i = [x for x in file_list if 'net_params_' + str(k + 1) + '_' in x]
        file_1 = file_i[0]
        file_i = file_i[0][len('net_params_' + str(k + 1) + '_'):]
        # get training loss
        trn_i = file_i[0:file_i.find('_')]
        file_i = file_i[len(trn_i) + 1:]
        # get validation loss
        val_i = file_i[0:file_i.find('.pth')]
        # store variables
        epoch.append(np.int(k + 1))
        trn.append(np.float(trn_i))
        val.append(np.float(val_i))
        list_sort.append(os.path.join(model_folder, file_1))
    # find the index of the best model with the lowest validation loss
    model_idx = np.argmin(val)
    # find the minimum validation loss
    val_model = val[model_idx]
    # find the corresponding training loss
    trn_model = trn[model_idx]
    model_path = list_sort[model_idx]
    return model_path, val_model, trn_model

# load model filepath
def load_model_path(exp_folder):
    # get file path
    f_list = os.listdir(exp_folder)
    model_list = []
    for kf in f_list:
        f_folder = os.path.join(exp_folder, kf)
        # get the best model
        model_path, val_model, trn_model = model_select(f_folder)
        model_list.append(model_path)
    return model_list

# get ROI model list
def load_model_roi(exp_folder):
    device = 'cuda:0'
    # load ROI model path
    model_roi_path_list = load_model_path('E:\\Landmark_HCMR\\model\\roi4')
    model_list = []
    for model_path in model_roi_path_list:
        # define model
        model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=1,
            out_channels=4,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2))
        model.load_state_dict(torch.load(model_path))
        model.to(device, dtype=torch.float)
        model.cuda()
        model_list.append(model)
    return model_list

# get landmark detection model list
def load_model_ldm(exp_folder):
    device = 'cuda:0'
    # load landmark detection model path
    model_roi_path_list = load_model_path('E:\\Landmark_HCMR\\model\\crop')
    model_list = []
    for model_path in model_roi_path_list:
        # define model
        model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=1,
            out_channels=14,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2))
        model.load_state_dict(torch.load(model_path))
        model.to(device, dtype=torch.float)
        model.cuda()
        model_list.append(model)
    return model_list

# save result 
def save_res(prd, ldm_file, csv_file):
    ldm = np.zeros((np.shape(prd)[0], np.shape(prd)[1], np.shape(prd)[2]))
    centroid_prd = np.zeros((np.shape(prd)[2], np.shape(prd)[3], 2))
    # get predicted annotated nii file
    for kf in range(np.shape(prd)[2]):
        ldm_i = np.zeros_like(prd[..., kf, 0])
        for kl in range(np.shape(prd)[3]):
            prd_in = prd[..., kf, kl]
            prd_in0 = prd_in - np.min(prd_in)
            prd_in1 = prd_in0 / np.max(prd_in0)
            prd_inb = prd_in1 >= 0.5
            prd_il = label(prd_inb)
            if np.max(prd_il) > 1:
                lab_n = np.unique(prd_il)
                n = []
                for kn in lab_n:
                    if kn == 0:
                        n.append(0)
                    elif kn > 0:
                        prd_iln = prd_il == kn
                        n.append(np.sum(prd_iln))
                prd_inb = prd_il == np.argmax(n)
            # compute centroid
            pi = centroid(prd_inb)
            centroid_prd[kf, kl, :] = pi
            prd_i_ = np.zeros_like(ldm_i)
            prd_i_[int(np.floor(pi[0])):int(np.floor(pi[0]) + 2),
                   int(np.floor(pi[1])):int(np.floor(pi[1]) + 2)] = 1
            ldm_i[prd_i_ == 1] = kl + 1
        ldm[..., kf] = ldm_i
        # predicted annotated nii file
    prd_ldm_nifti = nib.Nifti1Image(ldm, [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # save predicted annotated nii file
    nib.save(prd_ldm_nifti, ldm_file)
    # empty arrays 
    c_prd_x_flat = []
    c_prd_y_flat = []
    c_ref_x_flat = []
    c_ref_y_flat = []
    name_flat = []
    frame_flat = []
    point_flat = []
    # get reference and predicted landmark coordinates
    for kf in range(centroid_prd.shape[0]):
        for kl in range(centroid_prd.shape[1]):
            pth, file = os.path.split(ldm_file)
            name_flat.append(file[0:file.find('.nii.gz')])
            frame_flat.append(kf)
            point_flat.append(kl)
            c_prd_x_flat.append(centroid_prd[kf, kl, 0])
            c_prd_y_flat.append(centroid_prd[kf, kl, 1])
            c_ref_x_flat.append(centroid_prd[kf, kl, 0])
            c_ref_y_flat.append(centroid_prd[kf, kl, 1])
    # print reference and predicted landmarks to csv file
    d = {'col1': name_flat, 'col2': frame_flat, 'col3': point_flat, 'col4': c_prd_x_flat,
         'col5': c_prd_y_flat, 'col6': c_ref_x_flat, 'col7': c_ref_y_flat}
    df = pd.DataFrame(data=d)
    df.to_csv(csv_file)
    return

# process one case
def case_process(img_file, ldm_file, csv_file):
    # load the ROI models
    model_roi_list = load_model_roi('E:\\Landmark_HCMR\\model\\roi4')
    # load the landmark detection models
    model_ldm_list = load_model_ldm('E:\\Landmark_HCMR\\model\\crop')
    # img_file = 'E:\\Landmark_HCMR\\data\\Data26032021\\3CH\\HCMR_001_0001.nii.gz'
    # ldm_file = 'E:\\Landmark_HCMR\\data\\Data26032021\\3CH_prd\\HCMR_001_0001.nii.gz'
    # csv_file = 'E:\\Landmark_HCMR\\data\\Data26032021\\3CH_prd_csv\\HCMR_001_0001.csv'
    # get the predicted image
    img, prd = img_prd_tst(img_file, model_roi_list, model_ldm_list)
    # save result
    save_res(prd, ldm_file, csv_file)
    return

# process multiple cases
def batch_process():
    # data folder path
    data_folder = 'E:\\Landmark_HCMR\\data\\Data26032021'
    # get file paths
    img_folder = os.path.join(data_folder, '3CH')
    ldm_folder = os.path.join(data_folder, '3CH_prd')
    csv_folder = os.path.join(data_folder, '3CH_prd_csv')
    case_list = os.listdir(os.path.join(data_folder, '3CH_crop'))
    done_list = os.listdir(ldm_folder)
    tobe_list = [x for x in case_list if x not in done_list]
    for k in tobe_list:
        print(k)
        img_file = os.path.join(img_folder, k)
        ldm_file = os.path.join(ldm_folder, k)
        csv_file = os.path.join(csv_folder, k[0:k.find('.nii.gz')] + '.csv')
        try:
            # process each case
            case_process(img_file, ldm_file, csv_file)
        except ValueError:
            print('Not processed.')
    return
