# Data-Mining

Welcome to my Data Mining project repository! This project focuses on exploring and analyzing a high-dimensional dataset using various data mining techniques and classification models. The goal is to extract valuable insights, preprocess the data, reduce dimensionality, rebalance the dataset, and train a classification model for predicting the target variable.

# Project Overview
In this project, I followed a structured approach to analyze the dataset, including:
1) Exploratory Data Analysis (EDA): Investigating feature types, identifying missing values and outliers, assessing variable scales, and examining class balance. Visualizations were used to support data exploration and preprocessing decisions.
2) Data Preprocessing: Handling missing data, addressing outliers, and applying variable scaling techniques such as normalization. Preprocessing choices were informed by the EDA.
3) Feature Reduction: Employing feature selection methods based on correlations, mutual information, and gain ratio. Additionally, I explored feature extraction and selection using Principal Component Analysis (PCA) to reduce dimensionality.
4) Data Rebalancing: Addressing class imbalance through techniques like under- or over-sampling, synthetic data generation, or using modeling approaches that consider data imbalance explicitly, such as cost-sensitive learning.
5) Model Training: Fitting at least one classification model to predict the target variable based on the features. Multiple classification approaches were explored, including Random Forest, XGBoost, and others. The main performance metric used for model evaluation was the Area under the ROC curve (AUC).
6) Model Assessment: Reporting the estimated AUC score as a measure of the model's generalization performance. To ensure unbiased evaluation, the dataset was split into training and test subsets based on the grouping variable, Info_cluster.

# Key Findings
Based on the AUC-ROC scores, it can be concluded that Dataset 2 demonstrated excellent performance in terms of the model's ability to discriminate between positive and negative samples. With an impressive AUC-ROC score of 0.911, the model exhibited outstanding classification capabilities.

# Dependencies
The following dependencies are required to run the project:

Python 3.7+
Jupyter Notebook
Scikit-learn
Pandas
Numpy
Matplotlib
Seaborn
XGBoost

# Conclusion
This project provided valuable insights into the process of data mining and analysis. By applying various techniques such as exploratory data analysis, preprocessing, feature reduction, rebalancing, and classification modeling, I achieved impressive results in terms of predicting the target variable. The repository serves as a comprehensive guide and resource for anyone interested in understanding the intricacies of data mining and implementing similar projects.

Let's connect and discuss more about data mining and machine learning! Don't hesitate to reach out.
