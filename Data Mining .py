#!/usr/bin/env python
# coding: utf-8

# # Importing The Dataset

# In[1]:


import pandas as pd

data = pd.read_csv("df_training_level2.csv")
data.head(10) #display first 10 rows


# # Exploratory Data Analysis

# In[2]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split the data into training, testing, and validation sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(data, 
                                                                    data.Class, 
                                                                    test_size=0.2, 
                                                                    random_state=42,
                                                                    stratify=data['Info_cluster'])


# In[3]:


X_train_set.shape


# In[4]:


X_train_set.count()


# In[5]:


X_train_set.isnull().sum() #count missing values


# In[6]:


X_train_set.shape #print no of rows and columns


# In[7]:


# Create a bar plot to visualize the class distribution in the training set
plt.bar(X_train_set.Class.value_counts().index, X_train_set.Class.value_counts().values)

# Print the value counts of the 'Class' column in the training set
print(X_train_set.Class.value_counts())


# # Data Visualization

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


random_features = np.random.choice(X_train_set.columns, size=10, replace=False)
df_subset = X_train_set[random_features]


# In[10]:


# Plot a boxplot for each feature in the subset
plt.figure(figsize=(16,9))
sns.boxplot(data=df_subset)
plt.show()


# In[11]:


df_subset.hist(bins=20, figsize=(20,20))
plt.show()


# In[12]:


x_feature, y_feature = np.random.choice(df_subset.columns, size=2, replace=False)
sns.scatterplot(data=df_subset, x=x_feature, y=y_feature)
plt.show()

x_feature, y_feature = np.random.choice(df_subset.columns, size=2, replace=False)
sns.scatterplot(data=df_subset, x=x_feature, y=y_feature)
plt.show()


# In[13]:


# Plot a density plot for each feature in the subset
df_subset.plot(kind='density', subplots=True, layout=(5,2), sharex=False, figsize=(20,20))
plt.show()


# # Data Pre-Processing 

# In[14]:


# Drop any rows in the training set that contain missing values
X_train_set.dropna(inplace=True)
print(X_train_set.shape)


# In[15]:


# Print the column labels of the training set
X_train_set.columns


# In[16]:


main_data = X_train_set.drop(X_train_set.filter(regex='^Info_').columns, axis=1)


# In[17]:


# Define a function to calculate z-scores for each column of a DataFrame and identify outliers
def z_score(dataframe):
    threshold = 3

    mean = np.mean(dataframe)
    std = np.std(dataframe)
    z_scores = [(y - mean) / std for y in dataframe]
    return np.where(np.abs(z_scores) > threshold)


# In[18]:


# Create a new DataFrame `numerical_data` that only includes numeric columns of `main_data`
numerical_data = main_data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
numerical_data.head(5)

# Apply the z_score` function to each column of `numerical_data` and return the indices of outliers
outlier_indices = numerical_data.apply(lambda x: z_score(x))
outlier_indices


# In[19]:


# Extract the indices of the outliers and store them in a set
outlier_df = outlier_indices.iloc[0]
outlier_df = pd.DataFrame(outlier_df)
outlier_df.columns = ['Rows_to_exclude']

outlier_indices_final = outlier_df['Rows_to_exclude'].to_numpy()

outlier_indices_final = np.concatenate(outlier_indices_final, axis=0)

unique_outlier_indices = set(outlier_indices_final)
unique_outlier_indices


# In[20]:


# Create a boolean filter to exclude the rows with outliers and create a new DataFrame `data_without_outliers`
exclude_rows_filter = main_data.index.isin(unique_outlier_indices)
data_without_outliers = main_data[~exclude_rows_filter]


# In[24]:


print('The size of the original dataframe. ' + str(len(main_data)))
print('The length of the new dataframe after removing the outlier: ' + str(len(data_without_outliers)))
print('The variance between a new and an old dataframe.: ' + str(len(main_data) - len(data_without_outliers)))
print('The unique outlier lists length: ' + str(len(unique_outlier_indices)))


# In[25]:


data_without_outliers['Class'].value_counts()


# In[26]:


train_data = data_without_outliers
train_data.shape


# # Feature Reduction

# In[27]:


from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[28]:


feature_data = train_data.drop('Class', axis=1)
target_data = train_data['Class']


# In[29]:


# Create a RandomForestClassifier
random_forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)


# In[30]:


# Create the Boruta feature selection object
boruta_selector_obj = BorutaPy(random_forest, n_estimators='auto', verbose=2, random_state=1, max_iter=100)


# In[31]:


boruta_selector_obj.fit(feature_data.values, target_data.values)


# In[32]:


selected_boruta_features = feature_data.columns[boruta_selector_obj.support_]
print("Selected features using Boruta: ", selected_boruta_features)


# # Data Rebalancing 

# In[33]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler


# In[34]:


# Perform under-sampling to balance the data
under_sampler = RandomUnderSampler(sampling_strategy='majority')
X_train_balanced, y_train_balanced = under_sampler.fit_resample(feature_data[selected_boruta_features], target_data)


# In[35]:


# Investigate class balance after rebalancing
sns.countplot(x=y_train_balanced)
plt.show()
print(y_train_balanced.value_counts())


# In[36]:


print("Shape of df_without_outliers: ", data_without_outliers.shape)
X_selected_features = feature_data[selected_boruta_features]
print("Shape of X_boruta: ", X_selected_features.shape)


# In[37]:


print("Shape of X_train_balanced: ", X_train_balanced.shape)
print("Shape of y_train_balanced: ", y_train_balanced.shape)


# In[38]:


X = X_train_balanced
y = y_train_balanced


# # Data Modelling

# In[39]:


def evaluate_mod(estimator, features, target):
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    data_scaler = StandardScaler()
    X_train_scaled = data_scaler.fit_transform(X_train)
    X_test_scaled = data_scaler.transform(X_test)
    
    estimator.fit(X_train_scaled, y_train)
    y_pred_proba = estimator.predict_proba(X_test_scaled)[:, 1]
    
    auc_result = roc_auc_score(y_test, y_pred_proba)
    
    return auc_result, estimator, data_scaler


# In[40]:


# RandomForest for train
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_auc, rf_trained_model, rf_data_scaler = evaluate_mod(random_forest_clf, X, y)
print(f"RandomForest Classifier Mean AUC: {random_forest_auc:.3f}")


# In[41]:


y[y == -1] = 0 # XGBoost for train
xgb_clf = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_auc, xgb_trained_model, xgb_data_scaler = evaluate_mod(xgb_clf, X, y)

print(f"XGBoost Classifier Mean AUC: {xgb_auc:.3f}")


# # Evaluting Model For Test Dataset 

# In[42]:


X_test_clean = X_test_set.drop(X_test_set.filter(regex='^Info_').columns, axis=1)
X_test_clean.head()


# In[43]:


test_dataset = pd.concat([X_test_set, y_test_set], axis=1)
test_dataset = test_dataset.dropna()
test_dataset = test_dataset.drop(test_dataset.filter(regex='^Info_').columns, axis=1)
y_test_clean = test_dataset.pop('Class')
y_test_clean = y_test_clean.iloc[:,0]


# In[44]:


# XGBoost for test
X_test_set_selected = test_dataset[selected_boruta_features]
X_test_set_scaled = xgb_data_scaler.transform(X_test_set_selected)
y_test_clean[y_test_clean == -1] = 0
test_predictions = xgb_trained_model.predict(X_test_set_scaled)
roc_auc_score(y_test_clean, test_predictions)


# In[45]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_score, title='ROC curve'):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# In[46]:


plot_roc_curve(y_test_clean, test_predictions)


# In[47]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test_clean, test_predictions)
ConfusionMatrixDisplay(cm).plot()
plt.show()


# # Conclusion 
# 
# Through the process of data mining, I was able to find valuable insights and information related to the data. I extracted useful findings and trends from the data due to the knowledge we gained on the preprocessing and cleaning data, feature selection and engineering, and model training. The relevance of having high-quality data and understanding how it might potentially affect a model's performance and accuracy has been the subject of extensive study and research.
# 
# According to the AUC-ROC scores, it is possible to conclude that Dataset 2 demonstrates excellent performance in terms of the model's capacity to discriminate between positive and negative samples. This conclusion can be drawn since Dataset 2 has more information. The AUC-ROC score of 0.911 for Dataset 2 suggests that the model does a fantastic job of classifying the samples.
