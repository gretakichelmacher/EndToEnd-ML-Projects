# -*- coding: utf-8 -*-
"""
Created on ue Oct 31 08:31:03 2023

@author: Greta Kichelmacher
"""

## import the libraries
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st
import matplotlib.pyplot as plt
from category_encoders import LeaveOneOutEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.under_sampling import RandomUnderSampler


## download the data
df = pd.read_csv("C:/Users/User/Desktop/Applied Analytics/crunchbase/crunchbase/crunchbase.csv")
df.head()
# df.shape # (31984, 18)


################################################################################################

##### 2. DATA UNDERSTANDING AND PREPARATION ####


### A) Create a new target variable (success) with two values: 1 for success and 0 
#      for failure. Use the definition of startup success provided above to determine 
#      the value of the target variable

success_values = []


for index, row in filt_df.iterrows():
    if (row['ipo'] or row['is_acquired']) and not row['is_closed']:
        success_values.append(1)
    else:
        success_values.append(0)


filt_df['success'] = success_values



# occurrence of 0 and 1 values in filt_df
count1 = filt_df['success'].value_counts()

print("number of rows with 0:", count1[0]) # number of rows with 0: 8617
print("number of rows with 1:", count1[1]) # number of rows with 1: 772



## class imbalance plot
fig, ax = plt.subplots(figsize=(6.4, 4.8))
filt_df["success"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Class imbalance")
ax.set_xlabel("success")
ax.set_ylabel("Count")



### Combine the features related to the education levels of the founders 
#      (mba_degree, phd_degree, ms_degree, other_degree) into a new feature for 
#      the total number of degrees obtained by the founders (number_degrees).


filt_df['number_degrees'] = filt_df[['mba_degree', 'phd_degree', 'ms_degree', 'other_degree']].sum(axis=1)

filt_df.describe()



### B) Identify the numerical features in the dataset (including the new feature 
#      created in e) and show their correlations with one another and the target in 
#      a heatmap. Are any features highly correlated with one another? What does this tell you?


num_df = filt_df.select_dtypes(include=['number'])
cat_df = filt_df.select_dtypes(include=['object'])


#We create an heatmap to have a better visualization of the matrix
plt.figure(figsize=(12, 10))

fig, ax = plt.subplots()
sns.heatmap(num_df.corr(), annot=True, cmap=sns.color_palette('ch:s=.25,rot=-.25', as_cmap=True), ax=ax, annot_kws={"fontsize": 5})



### C) Identify the categorical features in the dataset. What will you need to do with the 
#      categorical features, before you can use them in the model? No code, just a description 
#      of what you need to do.

# look at the unique categories of each categorical variable
for column in cat_df.columns:
    print(f"{column}: {cat_df[column].unique()}")


### Some features have a large number of missing values. 

missing_counts = filt_df.isnull().sum()

missing_ratios = missing_counts / len(filt_df)

summary_df = pd.DataFrame({'Missing Counts': missing_counts, 'Missing Ratios': missing_ratios})


### J) Can we choose reasonable default values for the missing values or should any of 
#      these features be removed?

# distribution of numerical variables
filt_df.hist(bins=50, figsize=(20,15))



###############################################################################################

### 3. MODELLING ###

# create X and y
X = filt_df.copy()
X.drop(['company_id', 'ipo', 'is_acquired', 'is_closed', "mba_degree", "phd_degree", "ms_degree", 'success', 'other_degree'], axis = 1, inplace=True) 

y = filt_df['success']
y = pd.DataFrame(y)


# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) # The stratify=y ensures that the train and test splits have approximately the same percentage of samples for each class as present in the original dataset.


# occurrence of 0 and 1 values in y_train
count = y_train.value_counts()

print("number of rows with 0:", count[0]) # number of rows with 0: 6893
print("number of rows with 1:", count[1]) # number of rows with 1: 618

# print(X_train.shape) # (7511, 11)
# print(X_test.shape) # (1878, 11)
# print(y_train.shape) # (7511, 1)
# print(y_test.shape) # (1878, 1)


### B) Create a pipeline for pre-processing numerical and categorical features. This 
#      should also take care of missing values

# LeaveOneOut Encoder per 'category_code'
# create an instance for LeaveOneOut encoder
encoder = LeaveOneOutEncoder(cols=['category_code'])

# Encode the categorical variables
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

## Create the Pipelines 

# median_pipeline
median_pipeline = Pipeline([
    ('imputation_median', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler())
])


# zero_pipeline
zero_pipeline = Pipeline([
    ('imputation_zero', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])


# one_pipeline
one_pipeline = Pipeline([
    ('imputation_one', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=1)),
    ('scaler', StandardScaler())
])


# num_scaler
num_scaler = Pipeline([
    ('scaler', StandardScaler())
])


# onehot_categorical_pipeline
onehot_categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


## Combine the pipelines
preprocessor = ColumnTransformer([
    ('num_median', median_pipeline, ['products_number']),
    ('num_zero', zero_pipeline, ['average_funded', 'acquired_companies']),
    ('num_one', one_pipeline, ['offices']),
    ('num_scaler', num_scaler, ['total_rounds', 'average_participants', 'age', 'number_degrees', 'category_code']),
    ('cat_onehot', onehot_categorical_pipeline, ['country_code', 'state_code']),
    ]
)


# fit the preprocessor
X_train_transformed = preprocessor.fit_transform(X_train_encoded)
X_test_transformed = preprocessor.transform(X_test_encoded)


def transform_and_create_dataframe(X, preprocessor):
    transformed_array = preprocessor.transform(X)
    # Fetch the one-hot encoded column names
    onehot_columns = preprocessor.named_transformers_['cat_onehot'].named_steps['onehot'].get_feature_names_out(['country_code', 'state_code'])
    # Combine with original column names
    columns = ['products_number', 'average_funded', 'acquired_companies', 'offices', 'total_rounds', 'average_participants', 'age', 'number_degrees', 'category_code'] + list(onehot_columns)
    return pd.DataFrame(transformed_array, columns=columns)

X_train_transformed = transform_and_create_dataframe(X_train_encoded, preprocessor)
X_test_transformed = transform_and_create_dataframe(X_test_encoded, preprocessor)



### C) Create two models: one using logistic regression, the other using random forests. 

## LOGISTIC REGRESSION

# Set up the model with class_weight='balanced'
log_reg = LogisticRegression(random_state=42, class_weight='balanced')

# Define the parameters for GridSearchCV
param_grid_log = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs', 'liblinear']}

# Create the GridSearchCV object
grid_search_log = GridSearchCV(log_reg, param_grid_log, cv=3, n_jobs=-1)

# Perform the grid search
grid_search_log.fit(X_train_transformed, y_train.values.ravel())

# Get the results
best_params_log = grid_search_log.best_params_
best_score_log = grid_search_log.best_score_

# Create the model with the best parameters found and class_weight='balanced'
best_log_reg = LogisticRegression(**best_params_log, class_weight='balanced')

# Train the model with the training data
best_log_reg.fit(X_train_transformed, y_train.values.ravel())

# Make predictions on the training data
y_pred_train_log = best_log_reg.predict(X_train_transformed)

# Calculate performance metrics for training
accuracy_train_log = accuracy_score(y_train, y_pred_train_log)
precision_train_log = precision_score(y_train, y_pred_train_log)
recall_train_log = recall_score(y_train, y_pred_train_log)
f1_train_log = f1_score(y_train, y_pred_train_log)

# Make predictions on the test data
y_pred_test_log = best_log_reg.predict(X_test_transformed)

# Calculate performance metrics for testing
accuracy_test_log = accuracy_score(y_test, y_pred_test_log)
precision_test_log = precision_score(y_test, y_pred_test_log)
recall_test_log = recall_score(y_test, y_pred_test_log)
f1_test_log = f1_score(y_test, y_pred_test_log)

# Print the performance metrics
print("Training Set:")
print("Accuracy:", accuracy_train_log)
print("Precision:", precision_train_log)
print("Recall:", recall_train_log)
print("F1-Score:", f1_train_log)
print("\nTest Set:")
print("Accuracy:", accuracy_test_log)
print("Precision:", precision_test_log)
print("Recall:", recall_test_log)
print("F1-Score:", f1_test_log)

print(best_params_log) # {'C': 0.001, 'solver': 'lbfgs'}

## Best logistic regression model after GridSearchCV

# Create the model with the best parameters found and class_weight='balanced'
best_log_reg = LogisticRegression(C=0.001, solver='lbfgs', class_weight='balanced', random_state=42)

# Train the model with the training data
best_log_reg.fit(X_train_transformed, y_train.values.ravel())

# Make predictions on the test data
y_pred_test_log = best_log_reg.predict(X_test_transformed)

# Calculate performance metrics for testing
accuracy_test_log = accuracy_score(y_test, y_pred_test_log)
precision_test_log = precision_score(y_test, y_pred_test_log)
recall_test_log = recall_score(y_test, y_pred_test_log)
f1_test_log = f1_score(y_test, y_pred_test_log)

# Print the performance metrics for the test set
print("\nTest Set:")
print("Accuracy:", accuracy_test_log)
print("Precision:", precision_test_log)
print("Recall:", recall_test_log)
print("F1-Score:", f1_test_log)



## RANDOM FOREST

# Define the model
random_forest_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')

# Define the parameter grid
param_grid_random = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create GridSearchCV object
grid_search_random = GridSearchCV(random_forest_classifier, param_grid_random, n_jobs=-1, cv=3)

# perform the grid search
grid_search_random.fit(X_train_transformed, y_train.values.ravel())

# get the result
best_params_random = grid_search_random.best_params_
best_score_random = grid_search_random.best_score_

# Create the model with the best parameters found
best_model_random = RandomForestClassifier(**best_params_random)

# train the model with the training data
best_model_random.fit(X_train_transformed, y_train.values.ravel())

# make predictions on the training data
y_pred_train_random = best_model_random.predict(X_train_transformed)

# Calculate perfomance metrics for the training set
accuracy_train_random = accuracy_score(y_train.values.ravel(), y_pred_train_random)
precision_train_random = precision_score(y_train.values.ravel(), y_pred_train_random)
recall_train_random = recall_score(y_train.values.ravel(), y_pred_train_random)
f1_train_random = f1_score(y_train.values.ravel(), y_pred_train_random)

# make predictions on the test data
y_pred_test_random = best_model_random.predict(X_test_transformed)

# calculate performance metrics for the test set
accuracy_test_random = accuracy_score(y_test.values.ravel(), y_pred_test_random)
precision_test_random = precision_score(y_test.values.ravel(), y_pred_test_random)
recall_test_random = recall_score(y_test.values.ravel(), y_pred_test_random)
f1_test_random = f1_score(y_test.values.ravel(), y_pred_test_random)

# print the result
print("Training Set:")
print("Accuracy:", accuracy_train_random)
print("Precision:", precision_train_random)
print("Recall:", recall_train_random)
print("F1-Score:", f1_train_random)
print("\nTest Set:")
print("Accuracy:", accuracy_test_random)
print("Precision:", precision_test_random)
print("Recall:", recall_test_random)
print("F1-Score:", f1_test_random)


print(best_params_random) # {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}


# best Random Forest after GridSearchCV

# Given best parameters
best_params_random = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}

# Create the model with the best parameters found
best_model_random = RandomForestClassifier(**best_params_random, random_state=42, class_weight='balanced')

# Train the model with the training data
best_model_random.fit(X_train_transformed, y_train.values.ravel())

# Make predictions on the test data
y_pred_test_random = best_model_random.predict(X_test_transformed)

# Calculate performance metrics for the test set
accuracy_test_random = accuracy_score(y_test.values.ravel(), y_pred_test_random)
precision_test_random = precision_score(y_test.values.ravel(), y_pred_test_random)
recall_test_random = recall_score(y_test.values.ravel(), y_pred_test_random)
f1_test_random = f1_score(y_test.values.ravel(), y_pred_test_random)

# Print the results for the test set
print("\nTest Set:")
print("Accuracy:", accuracy_test_random)
print("Precision:", precision_test_random)
print("Recall:", recall_test_random)
print("F1-Score:", f1_test_random)



########################################################################################################

### 4. EVALUATION AND INTERPRETATION ###

### A) Which performance metrics are most suitable for evaluating your model to predict startup success? It helps to think about what performance metrics would matter most to an investor. Hint: It helps to read the article carefully.

## Feature importance with SHAP library (TAKES TIME TO RUN!)
import shap

# Inizialization shap explainer object
explainer = shap.Explainer(best_model_random.predict, X_test_transformed)


# Explain the individual predictions
shap_values = explainer(X_test_transformed)


# graph of total contribution of variables
fig = shap.plots.bar(shap_values, max_display=12, show = False)



### B) The dataset is umbalanced. There are many more failed startups than successful startups. 
#      Suggest two ways to deal with imbalanced data, and implement one of them.

# Apply Random UnderSampling to y_train
undersampler = RandomUnderSampler(random_state=42)
X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train_transformed, y_train)

# print(X_train_undersampled.shape) # (1236, 14)
# print(y_train_undersampled.shape) # (1236, 1)

# occurrence of 0 and 1 values in y_train
count3 = y_train_undersampled.value_counts()

print("number of rows with 0:", count3[0]) # number of rows with 0: 618
print("number of rows with 1:", count3[1]) # number of rows with 1: 618


# remove the extra features
variables_to_remove = ['products_number', 'acquired_companies', 'offices', 'total_rounds', 'country_code_NZL', 'state_code_California', 'state_code_other']

X_train_undersampled = X_train_undersampled.drop(columns=variables_to_remove, axis=1)
X_test_transformed = X_test_transformed.drop(columns=variables_to_remove, axis=1)


## LOGISTIC REGRESSION with UNDERSAMPLING

# Create the model with the best parameters found and class_weight='balanced'
best_log_reg_undersampled = LogisticRegression(C=0.001, solver='lbfgs', class_weight='balanced', random_state=42)


# Train the model with the training data
best_log_reg_undersampled.fit(X_train_undersampled, y_train_undersampled.values.ravel())


# Make predictions on the test data
y_pred_test_log_undersampled = best_log_reg_undersampled.predict(X_test_transformed)

# Calculate performance metrics for testing
accuracy_test_log_undersampled = accuracy_score(y_test, y_pred_test_log_undersampled)
precision_test_log_undersampled = precision_score(y_test, y_pred_test_log_undersampled)
recall_test_log_undersampled = recall_score(y_test, y_pred_test_log_undersampled)
f1_test_log_undersampled = f1_score(y_test, y_pred_test_log_undersampled)

# Print the performance metrics for the test set
print("\nTest Set:")
print("Accuracy:", accuracy_test_log_undersampled)
print("Precision:", precision_test_log_undersampled)
print("Recall:", recall_test_log_undersampled)
print("F1-Score:", f1_test_log_undersampled)





## RANDOM FOREST with UNDERSAMPLING

# Create the model with the best parameters found
best_model_random_undersampled = RandomForestClassifier(**best_params_random, random_state=42, class_weight='balanced')

# Train the model with the training data
best_model_random_undersampled.fit(X_train_undersampled, y_train_undersampled.values.ravel())

# Make predictions on the test data
y_pred_test_random_undersampled = best_model_random_undersampled.predict(X_test_transformed)

# Calculate performance metrics for the test set
accuracy_test_random_undersampled = accuracy_score(y_test.values.ravel(), y_pred_test_random_undersampled)
precision_test_random_undersampled = precision_score(y_test.values.ravel(), y_pred_test_random_undersampled)
recall_test_random_undersampled = recall_score(y_test.values.ravel(), y_pred_test_random_undersampled)
f1_test_random_undersampled = f1_score(y_test.values.ravel(), y_pred_test_random_undersampled)



# Print the results for the test set
print("\nTest Set:")
print("Accuracy:", accuracy_test_random_undersampled)
print("Precision:", precision_test_random_undersampled)
print("Recall:", recall_test_random_undersampled)
print("F1-Score:", f1_test_random_undersampled)




### C) Use stratified k-fold cross-validation to evaluate the model. Report the scores 
#      for each fold, and the averages.

# Stratified k-fold cross-validation for training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(**best_params_random, random_state=42, class_weight='balanced')
scores = cross_validate(model,  X_train_undersampled, y_train_undersampled.values.ravel(), cv=cv, n_jobs=-1,
 scoring=['accuracy', 'precision', 'recall', 'f1'])

# Performance scores for each fold
df_scores_training = pd.DataFrame(scores).transpose()
df_scores_training['mean'] = df_scores_training.mean(axis=1)
print(df_scores_training)


### D) For comparison, evaluate the model against the test dataset. Show the performance 
#      metrics in a heatmap.

# Stratified k-fold cross-validation for test data
model = RandomForestClassifier(**best_params_random, random_state=42, class_weight='balanced')
scores = cross_validate(model,  X_test_transformed, y_test.values.ravel(), cv=cv, n_jobs=-1,
 scoring=['accuracy', 'precision', 'recall', 'f1'])

# Performance scores for each fold
df_scores_test = pd.DataFrame(scores).transpose()
df_scores_test['mean'] = df_scores_test.mean(axis=1)
print(df_scores_test)


# Setting up the plot size
plt.figure(figsize=(12, 4))

# Plotting the heatmap for training data
plt.subplot(1, 2, 1)
sns.heatmap(df_scores_training, annot=True, cmap="YlGnBu", cbar=False, fmt='.3f')
plt.title("Training Data Metrics")

# Plotting the heatmap for test data
plt.subplot(1, 2, 2)
sns.heatmap(df_scores_test, annot=True, cmap="YlGnBu", cbar=False, fmt='.3f')
plt.title("Test Data Metrics")

plt.tight_layout()



##############################################################################################################


















