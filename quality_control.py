import pandas as pd
import os
import numpy as np

# get coordinates of landmarks in the five relevant frames 
def csv2x(csv_file):
    # load .xlsx file with all predicted landmarks for each case per frame
    #csv_file = "E:\\Landmark_HCMR\\data\\Data26032021\\3CH_prd_csv\\HCMR_001_0001.csv"
    case_file = "C:\\Users\\Manisha\\Desktop\\HCMR_LVOTO_All_Final.xlsx"
    
    # read data 
    csv_data = pd.read_csv(csv_file)
    case_data = pd.read_excel(case_file)
    # read columns into arrays   
    case_id = np.array(case_data['sitePatID'])
    echorest = np.array(case_data['echorest'])
    case_name = np.array(csv_data['col1'])
    frame_no = np.array(csv_data['col2'])
    x_vals = np.array(csv_data['col4'])
    y_vals = np.array(csv_data['col5'])
    # get case name
    case_no1 = case_name[0][5:8]
    case_no2 = case_name[0][9:13]
    separator = '-'
    case_no = case_no1 + separator + case_no2
    # get the LVOT pressure gradient
    case_idx = np.where(case_id == case_no)
    echorest_val = echorest[case_idx]
    echorest_class = ''
    
    # classify cases as obstructive, non-obstructive or NaN when there is missing data
    if echorest_val >= 30:
        echorest_class = 'obstructive'
    elif echorest_val < 30:
        echorest_class = 'non-obstructive'
    else:
        echorest_class = 'NaN'

    # create empty arrays
    dlv = []    
    label0 = []
    label1 = []
    label2 = []
    label3 = []
    label4 = []
    label5 = []
    label6 = []
    label7 = []
    label8 = []
    label9 = []
    label10 = []
    label11 = []
    label12 = []
    label13 = []

    frame_id_array = []
    frame_no_array = []
    
    case_name_array = []
    echorest_class_array = []
    echorest_val_array = []

    # loop over all frames 
    for kf in np.unique(frame_no):
        # get landmark coordinates for each label
        frame_idx = np.where(frame_no == kf)[0]
        landmark_idx0 = frame_idx[0]
        landmark_idx1 = frame_idx[1]
        landmark_idx2 = frame_idx[2]
        landmark_idx3 = frame_idx[3]
        landmark_idx4 = frame_idx[4]
        landmark_idx5 = frame_idx[5]
        landmark_idx6 = frame_idx[6]
        landmark_idx7 = frame_idx[7]
        landmark_idx8 = frame_idx[8]
        landmark_idx9 = frame_idx[9]
        landmark_idx10 = frame_idx[10]
        landmark_idx11 = frame_idx[11]
        landmark_idx12 = frame_idx[12]
        landmark_idx13 = frame_idx[13]
        
        x0 = x_vals[landmark_idx0]
        y0 = y_vals[landmark_idx0]
        x1 = x_vals[landmark_idx1]
        y1 = y_vals[landmark_idx1]
        x2 = x_vals[landmark_idx2]
        y2 = y_vals[landmark_idx2]
        x3 = x_vals[landmark_idx3]
        y3 = y_vals[landmark_idx3]
        x4 = x_vals[landmark_idx4]
        y4 = y_vals[landmark_idx4]
        x5 = x_vals[landmark_idx5]
        y5 = y_vals[landmark_idx5]
        x6 = x_vals[landmark_idx6]
        y6 = y_vals[landmark_idx6]
        x7 = x_vals[landmark_idx7]
        y7 = y_vals[landmark_idx7]
        x8 = x_vals[landmark_idx8]
        y8 = y_vals[landmark_idx8]
        x9 = x_vals[landmark_idx9]
        y9 = y_vals[landmark_idx9]
        x10 = x_vals[landmark_idx10]
        y10 = y_vals[landmark_idx10]
        x11 = x_vals[landmark_idx11]
        y11 = y_vals[landmark_idx11]
        x12 = x_vals[landmark_idx12]
        y12 = y_vals[landmark_idx12]
        x13 = x_vals[landmark_idx13]
        y13 = y_vals[landmark_idx13]
        
        cp0 = [x0,y0]
        cp1 = [x1,y1]
        cp2 = [x2,y2]
        cp3 = [x3,y3]
        cp4 = [x4,y4]
        cp5 = [x5,y5]
        cp6 = [x6,y6]
        cp7 = [x7,y7]
        cp8 = [x8,y8]
        cp9 = [x9,y9]
        cp10 = [x10,y10]
        cp11 = [x11,y11]
        cp12 = [x12,y12]
        cp13 = [x13,y13]
        
        label0.append(cp0)
        label1.append(cp1)
        label2.append(cp2)
        label3.append(cp3)
        label4.append(cp4)
        label5.append(cp5)
        label6.append(cp6)
        label7.append(cp7)
        label8.append(cp8)
        label9.append(cp9)
        label10.append(cp10)
        label11.append(cp11)
        label12.append(cp12)
        label13.append(cp13)
        
        # calculate the LV width in all frames
        dlv.append(np.linalg.norm(np.array(cp8) - np.array(cp9)))
        
    # find index of five frames (mid systole -2, mid systole, mid systole +2, end systole and end diastole) based on the LV width
    # end systolic frame -> find frame with minimum LV width
    es = np.argmin(dlv)
    # end diastolic frame -> find frame with maximum LV width
    ed = np.argmax(dlv[es:]) + es
    # mid systolic frame -> find frame halfway between first frame and mid systole
    ms = int(np.round(es/2))
    # find two frame before and after mid systole
    ms_2before = ms - 2
    ms_2after = ms + 2
    
    # get landmark coordinates for the five frames only
    label_0_5 = [label0[ms_2before],label0[ms],label0[ms_2after],label0[es],label0[ed]]
    label_1_5 = [label1[ms_2before],label1[ms],label1[ms_2after],label1[es],label1[ed]]
    label_2_5 = [label2[ms_2before],label2[ms],label2[ms_2after],label2[es],label2[ed]]
    label_3_5 = [label3[ms_2before],label3[ms],label3[ms_2after],label3[es],label3[ed]]
    label_4_5 = [label4[ms_2before],label4[ms],label4[ms_2after],label4[es],label4[ed]]
    label_5_5 = [label5[ms_2before],label5[ms],label5[ms_2after],label5[es],label5[ed]]
    label_6_5 = [label6[ms_2before],label6[ms],label6[ms_2after],label6[es],label6[ed]]
    label_7_5 = [label7[ms_2before],label7[ms],label7[ms_2after],label7[es],label7[ed]]
    label_8_5 = [label8[ms_2before],label8[ms],label8[ms_2after],label8[es],label8[ed]]
    label_9_5 = [label9[ms_2before],label9[ms],label9[ms_2after],label9[es],label9[ed]]
    label_10_5 = [label10[ms_2before],label10[ms],label10[ms_2after],label10[es],label10[ed]]
    label_11_5 = [label11[ms_2before],label11[ms],label11[ms_2after],label11[es],label11[ed]]
    label_12_5 = [label12[ms_2before],label12[ms],label12[ms_2after],label12[es],label12[ed]]
    label_13_5 = [label13[ms_2before],label13[ms],label13[ms_2after],label13[es],label13[ed]]

    # frame id and number
    frame_id_array = np.array(['ms_2before','ms','ms_2after','es','ed'])
    frame_no_array = np.array([ms_2before,ms,ms_2after,es,ed])
    # case name
    case_name_array = np.array([case_name[0],case_name[0],case_name[0],case_name[0],case_name[0]])
    # case classification
    echorest_class_array = np.array([echorest_class,echorest_class,echorest_class,echorest_class,echorest_class])
    # LVOT pressure gradient
    echorest_val_array = np.repeat(echorest_val,5)
    # make array with all case data 
    final_data = np.concatenate((case_name_array,echorest_class_array,echorest_val_array,frame_id_array,frame_no_array),axis=0)
    final_data = np.reshape(final_data, (5, 5), order='F')
    
    return final_data,[label_0_5,label_1_5,label_2_5,label_3_5,label_4_5,label_5_5,label_6_5,label_7_5,label_8_5,label_9_5,label_10_5,label_11_5,label_12_5,label_13_5]


# get all outlier cases 
def all_cases():
    # load folder containing csv files with case data and predicted landmark coordinates
    case_folder = "C:\\Users\\Manisha\\Desktop\\Data2000\\3CH_prd_csv"
    file_lst = os.listdir(case_folder)
    
    # create empty arrays
    norm_dist_all = []
    case_all = []
    
    # loop over all cases
    for csv_file in file_lst:
        print(csv_file)
        # compute landmark coordinates in the five frames of interest
        final_data, label = csv2x(os.path.join(case_folder,csv_file))
        # get the classification
        echo_rest_class = final_data[0][1]
        if echo_rest_class =='NaN':
            print('skip')
        else:
            # calculate centre of mass of landmark coordinates
            mean_label = np.mean(np.reshape(np.array(label),(70,2)),axis=0)
            # compute LV length in end diastolic frame
            lv_length = np.linalg.norm(np.array(label[10][4])-np.array(label[11][4]))
            label_flat = np.reshape(np.array(label),(70,2))
            # compute distance between landmark coordinates and the centre of mass
            label_dist = np.linalg.norm(label_flat-mean_label,axis=1)
            # normalise distance by the LV length
            label_norm = label_dist/lv_length
            # store normalised distances
            norm_dist_all.append(label_norm)
            case_all.append(final_data[0][0])
            
    norm_dist_array = np.array(norm_dist_all) 
    
    # create empty array
    outlier_idx = []
    # loop over all landmarks in all frames for each case
    for k in range(70):
        # get normalised distances
        dist_i = norm_dist_array[:,k]
        # compute the mean distance
        mean_i = np.mean(dist_i)
        # compute the standard deviation of the mean distance
        std_i = np.std(dist_i)
        # compute upper and lower bounds (3 standard deviations of the mean)
        upper_idx = np.where(dist_i > mean_i + 3*std_i)
        lower_idx = np.where(dist_i < mean_i - 3*std_i)
        
        # exclude cases where one or more landmarks lie outside 3 standard deviations of the mean distance between the landmark and centre of mass 
        if len(upper_idx[0]) > 0:
            for ku in upper_idx[0]:
                outlier_idx.append(ku)
        if len(lower_idx[0]) > 0:
            for kl in lower_idx[0]:
                outlier_idx.append(kl)
    
    # find the outlier case names            
    outlier = np.unique(np.array(outlier_idx))
    case_outlier = []
    for idx in outlier:
        case_outlier.append(case_all[np.int(idx)])
    print(case_outlier)
    
    return 










