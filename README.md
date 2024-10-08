## HCM LVOTO Landmark Analysis AI Tool

### Project Overview
The aim of the project was to create a pipeline that can automatically detect anatomical landmarks in cardiac magnetic resonance (CMR) images using deep learning and predict the presence of left ventricular outflow tract obstruction (LVOTO) in hypertrophic cardiomyopathy (HCM) patients using machine learning models. 

### Repository Contents

- ‘heatmap.py’: Python code to generate heatmaps from annotated .nii files 

- ‘prep_train_roi’ : Python code used to train the region of interest (ROI) network

- ‘prep_train_ld’: Python code used to train the landmark detection network

- ‘prep_test_5_fold.py’: Python code used to test the ROI and landmark detection networks

- ‘landmark_nn.py’: Python code used for inferencing to predict anatomical landmarks 

- ‘save_label_csv.py’: Python code used to generate .csv files with landmark coordinates extracted from annotated .nii files

- ‘quality_control.py’: Python code used to detect outlier cases based on the predicted landmarks for quality control

- ‘get_distances.py’: Python code to compute distances between predicted landmarks in the five frames of interest

- ‘pls_regression.py’: Python code for the PLS regression model to predict LVOTO based on the computed distances between landmarks

- ‘logistic_regression.py’: Python code for the Logistic regression classifier to predict LVOTO based on the computed distances between landmarks

### Documentation
For more details about the methodology, please refer to the following paper:
- Machine learning evaluation of LV outflow obstruction in hypertrophic cardiomyopathy using three-chamber cardiovascular magnetic resonance M Sahota, SR Saraskani, H Xu, L Li, AW Majeed, U Hermida, S Neubauer, M Desai, W Weintraub, P Desvigne-Nickens, J Schulz-Menger, RY Kwong, CM Kramer, AA Young and P Lamata on behalf of the HCMR investigators (https://link.springer.com/article/10.1007/s10554-022-02724-7#Sec2)
