import pandas as pd
import os
import numpy as np

# get coordinates of landmarks in the five relevant frames 
def csv2x(csv_file):
    # load .xlsx file with all predicted landmarks for each case per frame
    #csv_file = "C:\\Users\\Manisha\\Desktop\\Data2000\\3CH_prd_csv\\HCMR_001_0001.csv"
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
    BS = []
    IVS = []
    AML = []
    PtoIVS = []
    LVwidth = []
    LVlength = []
    LVOT = []
    AMLtoBS = []
    dlv = []
    
    distance_BST = []
    distance_IVS = []
    distance_AML = []
    distance_PtoIVS = []
    distance_LVwidth = []
    distance_LVlength = []
    distance_LVOT = []
    distance_AMLtoBS = []

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
        
        # compute distance between landmarks 
        BS.append(np.linalg.norm(np.array(cp0) - np.array(cp1)))
        IVS.append(np.linalg.norm(np.array(cp2) - np.array(cp3)))
        AML.append(np.linalg.norm(np.array(cp4) - np.array(cp5)))
        PtoIVS.append(np.linalg.norm(np.array(cp6) - np.array(cp7)))
        LVwidth.append(np.linalg.norm(np.array(cp8) - np.array(cp9)))
        LVlength.append(np.linalg.norm(np.array(cp10) - np.array(cp11)))
        LVOT.append(np.linalg.norm(np.array(cp12) - np.array(cp13)))
        AMLtoBS.append(np.linalg.norm(np.array(cp5) - np.array(cp0)))
        
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
    
    # get distances for each frame
    BS_es = BS[es]
    BS_ed = BS[ed]
    BS_ms = BS[ms]
    BS_ms_2before = BS[ms_2before]
    BS_ms_2after = BS[ms_2after]
    
    IVS_es = IVS[es]
    IVS_ed = IVS[ed]
    IVS_ms = IVS[ms]
    IVS_ms_2before = IVS[ms_2before]
    IVS_ms_2after = IVS[ms_2after]
    
    AML_es = AML[es]
    AML_ed = AML[ed]
    AML_ms = AML[ms]
    AML_ms_2before = AML[ms_2before]
    AML_ms_2after = AML[ms_2after]
    
    PtoIVS_es = PtoIVS[es]
    PtoIVS_ed = PtoIVS[ed]
    PtoIVS_ms = PtoIVS[ms]
    PtoIVS_ms_2before = PtoIVS[ms_2before]
    PtoIVS_ms_2after = PtoIVS[ms_2after]
    
    LVwidth_es = LVwidth[es]
    LVwidth_ed = LVwidth[ed]
    LVwidth_ms = LVwidth[ms]
    LVwidth_ms_2before = LVwidth[ms_2before]
    LVwidth_ms_2after = LVwidth[ms_2after]
    
    LVlength_es = LVlength[es]
    LVlength_ed = LVlength[ed]
    LVlength_ms = LVlength[ms]
    LVlength_ms_2before = LVlength[ms_2before]
    LVlength_ms_2after = LVlength[ms_2after]
    
    LVOT_es = LVOT[es]
    LVOT_ed = LVOT[ed]
    LVOT_ms = LVOT[ms]
    LVOT_ms_2before = LVOT[ms_2before]
    LVOT_ms_2after = LVOT[ms_2after]
    
    AMLtoBS_es = AMLtoBS[es]
    AMLtoBS_ed = AMLtoBS[ed]
    AMLtoBS_ms = AMLtoBS[ms]
    AMLtoBS_ms_2before = AMLtoBS[ms_2before]
    AMLtoBS_ms_2after = AMLtoBS[ms_2after]
    
    # make an array with measurements from all five frames for each distance    
    distance_BST = np.array([BS_ms_2before,
                             BS_ms,
                             BS_ms_2after,
                             BS_es,
                             BS_ed])
    
    distance_IVS = np.array([IVS_ms_2before,
                             IVS_ms,
                             IVS_ms_2after,
                             IVS_es,
                             IVS_ed])
    
    distance_AML = np.array([AML_ms_2before,
                             AML_ms,
                             AML_ms_2after,
                             AML_es,
                             AML_ed])
    
    distance_PtoIVS = np.array([PtoIVS_ms_2before,
                                PtoIVS_ms,
                                PtoIVS_ms_2after,
                                PtoIVS_es,
                                PtoIVS_ed])
    
    distance_LVwidth = np.array([LVwidth_ms_2before,
                                 LVwidth_ms,
                                 LVwidth_ms_2after,
                                 LVwidth_es,
                                 LVwidth_ed])
    
    distance_LVlength = np.array([LVlength_ms_2before,
                                  LVlength_ms,
                                  LVlength_ms_2after,
                                  LVlength_es,
                                  LVlength_ed])
    
    distance_LVOT = np.array([LVOT_ms_2before,
                              LVOT_ms,
                              LVOT_ms_2after,
                              LVOT_es,
                              LVOT_ed])
    
    distance_AMLtoBS = np.array([AMLtoBS_ms_2before,
                                 AMLtoBS_ms,
                                 AMLtoBS_ms_2after,
                                 AMLtoBS_es,
                                 AMLtoBS_ed])

    # frame id and number
    frame_id_array = np.array(['ms_2before','ms','ms_2after','es','ed'])
    frame_no_array = np.array([ms_2before,ms,ms_2after,es,ed])
    # case name
    case_name_array = np.array([case_name[0],case_name[0],case_name[0],case_name[0],case_name[0]])
    # case classification
    echorest_class_array = np.array([echorest_class,echorest_class,echorest_class,echorest_class,echorest_class])
    # LVOT pressure gradient
    echorest_val_array = np.repeat(echorest_val,5)

    # make array with all case data and distances
    final_data = np.concatenate((case_name_array,echorest_class_array,echorest_val_array,frame_id_array,frame_no_array,distance_BST,distance_IVS,distance_AML,distance_PtoIVS,distance_LVwidth,distance_LVlength,distance_LVOT,distance_AMLtoBS),axis=0)
    final_data = np.reshape(final_data, (5, 13), order='F')
    
    return final_data
    
    
def get_distances():
    # load folder containing csv files with case data and predicted landmark coordinates
    data_folder = "E:\\Landmark_HCMR\\data\\Data26032021\\3CH_prd_csv"
    data_lst = os.listdir(data_folder)

    # create empty arrays
    final_case_name = []
    final_echorest_class = []
    final_echorest_val = []
    final_frame_id = []
    final_frame_no = []
    final_distance_BST = []
    final_distance_IVS = []
    final_distance_AML = []
    final_distance_PtoIVS = []
    final_distance_LVwidth = []
    final_distance_LVlength = []
    final_distance_LVOT = []
    final_distance_AMLtoBS = []
    data_all = [final_case_name,final_echorest_class,final_echorest_val,final_frame_id,final_frame_no,final_distance_BST,
                final_distance_IVS,final_distance_AML,final_distance_PtoIVS,final_distance_LVwidth,final_distance_LVlength,
                final_distance_LVOT,final_distance_AMLtoBS]
    # loop over all cases and compute distances in the five frames of interest 
    for data_file in data_lst:
        print(data_file[0:data_file.find('.csv')])
        data_path = os.path.join(data_folder,data_file)
        final_data_all = csv2x(data_path)
        for kf in range(len(final_data_all)):
            for ki in range(len(final_data_all[kf])):
                data_all[ki].append(final_data_all[kf,ki])
            

    # print csv file with all case data and distances  
    d = {'case_name': data_all[0], 'obstructive/non-obstructive': data_all[1], 'echorest': data_all[2], 'frame_id': data_all[3],
         'frame_no': data_all[4], 'BST (pixels)': data_all[5], 'IVS (pixels)': data_all[6],'AML (pixels)': data_all[7],'PtoIVS (pixels)': data_all[8],'LV Width (pixels)': data_all[9],
         'LV Length (pixels)': data_all[10],'LVOT (pixels)': data_all[11],'AMLtoBS (pixels)':data_all[12]}
    
    df = pd.DataFrame(data=d)
    output_file = "C:\\Users\\Manisha\\Desktop\\Data2000_Distances.csv"
    df.to_csv(output_file)    
    
    return 
