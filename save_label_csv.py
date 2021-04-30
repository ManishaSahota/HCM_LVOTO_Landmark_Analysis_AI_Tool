import numpy as np
import os
import pandas as pd
import nibabel as nib

import warnings
warnings.filterwarnings('ignore')

TYPEOF_LABEL=14

# get csv files with landmark coordinates from annotated nii files
def nii2csv(nii_file, case_name, save_folder=''):
    # nii_file is the abs_path of a nii file
    if os.path.isfile(nii_file) and 'nii' in nii_file:
        # load image nii file
        im=nib.load(nii_file)
        # get image data
        dat=im.get_data()
        # get number of frames
        nb_frames=dat.shape[-1]
        # csv file headers
        csv_mat=[['case', 'label', 'frame', 'imgx', 'imgy']]
        
        # loop over frames and extract landmark coordinates for each label
        for i in range(nb_frames):
            temp_img=dat[:,:,i]
            for j in range(1,TYPEOF_LABEL+1):
                coords=np.where(temp_img==j)
                if len(coords):
                    nb_labels=len(coords[0])
                    for k in range(nb_labels):
                        # store coordinates
                        temp_mat = [[case_name, str(j),str(i),str(coords[0][k]), str(coords[1][k])]]
                        csv_mat = np.append(csv_mat, temp_mat, axis=0)
        # print coordinates to csv files 
        csv_pd = pd.DataFrame(csv_mat)
        save_path=os.path.join(save_folder,case_name+'_Manisha.csv')
        csv_pd.to_csv(save_path, sep=',', header=False, index=False)
        
    else:
        print ('wrong file: '+case_name)
 

# read nii files in a nii_folder, and save them in a csv_folder
nii_folder='annotated_files'
csv_folder='csvs'

nii_files=os.listdir(nii_folder)
# loop over all nii files in folder, extract landmark coordinates and print csv file
for nii_file in nii_files:
    nii_file_path=os.path.join(nii_folder,nii_file)
    case_name=nii_file[:13]
    print(case_name)
    nii2csv(nii_file_path, case_name, csv_folder)