import numpy as np
import scipy.io as sio
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc 
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


# plot ROC curve using the false positive and true positive rates
def plot_roc_cur(fper, tper,auc):  
    plt.plot(fper, tper, color='orange', label='ROC (AUC = %0.2f)' % (auc))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    return

# randomly divide cases into 10 folds for 10 fold cross valdiation 
def idx_gen():
    idx = list(range(1780))
    
    idx1 = random.sample(idx,178)
    idx1_ = [x for x in idx if x not in idx1]
    
    idx2 = random.sample(idx1_,178)
    idx2_ = [x for x in idx1_ if x not in idx2]
    
    idx3 = random.sample(idx2_,178)
    idx3_ = [x for x in idx2_ if x not in idx3]
    
    idx4 = random.sample(idx3_,178)
    idx4_ = [x for x in idx3_ if x not in idx4]

    idx5 = random.sample(idx4_,178)
    idx5_ = [x for x in idx4_ if x not in idx5]
    
    idx6 = random.sample(idx5_,178)
    idx6_ = [x for x in idx5_ if x not in idx6]
    
    idx7 = random.sample(idx6_,178)
    idx7_ = [x for x in idx6_ if x not in idx7]
    
    idx8 = random.sample(idx7_,178)
    idx8_ = [x for x in idx7_ if x not in idx8]
    
    idx9 = random.sample(idx8_,178)
    idx10 = [x for x in idx8_ if x not in idx9]
    
    #sio.savemat('idx_10_fold_no_outliers.mat',{'idx1':idx1,'idx2':idx2,'idx3':idx3,'idx4':idx4,'idx5':idx5,'idx6':idx6,
    #                              'idx7':idx7,'idx8':idx8,'idx9':idx9,'idx10':idx10})
    
    return 

# get case index 
def idx_init(i):
    idx = list(range(1780))
    idx_data = sio.loadmat('idx_10_fold_no_outliers.mat')
    idx_tst = idx_data['idx'+str(i+1)][0]
    idx_trn = [x for x in idx if x not in idx_tst]
    
    return idx_tst, idx_trn

# PLS regressor 
def pls():
    # load csv file with case data and distances
    csv_file = "C:\\Users\\Manisha\\Desktop\\Data2000\\Data_Outliers_Removed.xlsx"  
    # get distances
    csv_data = pd.read_excel(csv_file)
    echorest = np.array(csv_data['echorest']).tolist()    
    BST = np.array(csv_data['BST (mm)']).tolist() 
    IVS = np.array(csv_data['IVS (mm)']).tolist() 
    AML = np.array(csv_data['AML (mm)']).tolist() 
    PtoIVS = np.array(csv_data['PtoIVS (mm)']).tolist() 
    LVwidth = np.array(csv_data['LV Width (mm)']).tolist() 
    LVlength = np.array(csv_data['LV Length (mm)']).tolist() 
    LVOT = np.array(csv_data['LVOT (mm)']).tolist() 
    AMLtoBS = np.array(csv_data['AMLtoBS (mm)']).tolist()
    AML_LVWidth = np.array(csv_data['AML/LV Width']).tolist() 
    AML_LVOT = np.array(csv_data['AML/LVOT diameter']).tolist() 
    
    X = []
    Y = []
    idx = np.array(list(range(1780)))
    
    i = 0
    
    # make array with all distances from each frame
    while i < (len(echorest)):
        arr_x = np.array([BST[i],BST[i+1],BST[i+2],BST[i+3],BST[i+4],
                          IVS[i],IVS[i+1],IVS[i+2],IVS[i+3],IVS[i+4],
                          AML[i],AML[i+1],AML[i+2],AML[i+3],AML[i+4],
                          PtoIVS[i],PtoIVS[i+1],PtoIVS[i+2],PtoIVS[i+3],PtoIVS[i+4],
                          LVwidth[i],LVwidth[i+1],LVwidth[i+2],LVwidth[i+3],LVwidth[i+4],
                          LVlength[i],LVlength[i+1],LVlength[i+2],LVlength[i+3],LVlength[i+4],
                          LVOT[i],LVOT[i+1],LVOT[i+2],LVOT[i+3],LVOT[i+4],
                          AMLtoBS[i],AMLtoBS[i+1],AMLtoBS[i+2],AMLtoBS[i+3],AMLtoBS[i+4],
                          AML_LVWidth[i],AML_LVWidth[i+1],AML_LVWidth[i+2],AML_LVWidth[i+3],AML_LVWidth[i+4],
                          AML_LVOT[i],AML_LVOT[i+1],AML_LVOT[i+2],AML_LVOT[i+3],AML_LVOT[i+4]])
        
        X.append(arr_x)

        # get the LVOT pressure gradient              
        arr_y = np.array(echorest[i])
        i = i + 5
        # assign to class (non-obstructive = 0, obstructive = 1)
        if arr_y >= 30:
            label = 1
        else:
            label = 0
        Y.append(label)
    
    
    Y_pred = np.zeros((1780,))
    
    for fold in range(10):
        # get training set and test set index
        idx_tst,idx_trn = idx_init(fold)
        
        # get training set
        Y_trn = []
        X_trn = []
        for idx in idx_trn:
            Y_trn.append(Y[idx])
            X_trn.append(X[idx])
            
        # create PLS Regression model
        pls = PLSRegression(n_components=1)
        # fit PLS Regression model to training data
        pls.fit(X_trn,Y_trn)
        
        # get test set
        Y_tst = []
        X_tst = []
        for idx in idx_tst:
            Y_tst.append(Y[idx])
            X_tst.append(X[idx])
    
        # get predictions on test set using PLS regression model
        y_pred = pls.predict(X_tst)
        
        for idx in range(len(idx_tst)):
            Y_pred[idx_tst[idx]] = y_pred[idx]
            
            
    # evaluate model performance by comparing ground truth and predictions
    # compute accuracy
    error = np.abs(np.round(Y_pred) - Y)
    error_rate = np.sum(error)/1780 
    error_rate_final = 1 - error_rate    
    print("accuracy = %0.3f" %error_rate_final)
    
    # compute specificity and sensitivity
    tn, fp, fn, tp = confusion_matrix(np.round(Y_pred), Y).ravel()
    specificity1 = tn / (tn+fp)
    sensitivity1 = tp / (tp+fn)
    print("specificity = %0.3f" %specificity1)
    print("sensitivity = %0.3f" %sensitivity1)

    # plot ROC curve 
    fper, tper, thresholds = roc_curve(Y, Y_pred,pos_label=1,drop_intermediate=False) 
    roc_auc1 = auc(fper, tper)
    print("AUC = %0.3f" %roc_auc1)
    plot_roc_cur(fper, tper,roc_auc1)
         
    return 



