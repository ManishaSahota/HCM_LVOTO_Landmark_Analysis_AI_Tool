import numpy as np
import os
import nibabel as nib
from scipy.ndimage import gaussian_filter
import scipy.io as sio

# folder path
img_folder = 'C:/Manisha/Desktop/landmark_detection_HCM/data/image_r'
nii_folder = 'C:/Manisha/Desktop/landmark_detection_HCM/data/manisha_r'

# generate heatmaps from annotated nii files

# get nii files
nii_files = os.listdir(nii_folder)
for nii_file in nii_files:
    nii_file_path = os.path.join(nii_folder, nii_file)
    case_name = nii_file[:13]
    img_file_path = os.path.join(img_folder, case_name + '.nii.gz')
    if os.path.isfile(nii_file_path) and 'nii' in nii_file and os.path.isfile(img_file_path):
        # load image nii file 
        img_file = nib.load(img_file_path)
        # get image data
        img = img_file.get_data()
        # load annotated nii file 
        landmark_file = nib.load(nii_file_path)
        # get annotated landmark data
        data = landmark_file.get_data()
        # get annotated frame numbers
        frames = np.unique(np.where(data)[2])
        nb_frames = len(frames)
        # number of landmarks
        nb_points = 14
        
        lab = np.zeros((np.shape(data)[0], np.shape(data)[1], nb_frames, nb_points))
        # get all landmarks in all frames 
        for kf_idx in range(nb_frames):
            kf = frames[kf_idx]
            data_f = data[..., kf]
            lab_f = np.transpose(np.array([np.zeros_like(data_f)] * nb_points), [1, 2, 0])
            for kp in range(nb_points):
                lab_fp = lab_f[..., kp]
                lab_fp[np.where(data_f == kp + 1)] = 1e6
                # apply guassian filter kernel of width sigma
                lab_fp = gaussian_filter(lab_fp, sigma=6)  
                # normalise intensity
                if np.max(lab_fp) > 0:
                    lab_fp = lab_fp / np.max(lab_fp)
                lab_f[..., kp] = lab_fp
            lab[..., kf_idx, :] = lab_f
        # save heatmaps 
        sio.savemat(os.path.join(nii_folder, case_name + '_heatmap_6.mat'), {'x': img[..., frames], 'y': lab})