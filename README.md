# BEng_HCMR_Project

The BEng_HCMR_Project repository contains the full code behind the deep learning pipeline for automatic landmark detection of anatomical features from CMR images and the predictive machine learning models for determining obstruction in HCM. 

The repository contains the following files:

‘heatmap.py’: generates heatmaps from annotated .nii files 

‘prep_train_roi’ : training code for the ROI network

‘prep_train_ld’: training code for the landmark detection network

‘prep_test_5_fold.py’: testing code for the ROI and landmark detection networks

‘landmark_nn.py’: inferencing code for obtaining model landmark predictions 

‘save_label_csv.py’: generates csv files with landmark coordinates extracted from annotated .nii files

‘quality_control.py’: detects outlier cases based on the predicted landmarks for quality control

‘get_distances.py’: determines the five frames of interest and computes distances between the predicted landmarks

‘pls_regression.py’: PLS regression model for prediction of obstruction using the predicted distances

‘logistic_regression.py’: Logistic regression classifier for prediction of obstruction using the predicted distances
