#!/usr/bin/env python
# coding: utf-8

# #            DIABETICS PREDICTION USING LOGISTIC REGRESSION 
#                         PREDICTING WHETHER A PERSON HAS DIABETES OR NOT 
# 	
#   

# # Framing the Problem
# 
# The objective is to Predict whether a person has diabetes or not based on a number of labels or features: 
# this includes: 
#         
#         Pregnancy : number of times pregnant
# 	        
#         Glucose: glucose level
# 		
#         Blood Pressure:  Diastolic blood pressure (mm Hg)
# 		
#         Skin Thickness:  Triceps skin fold thickness (mm)
# 		
#         Insulin: 2-Hour serum insulin (mu U/ml)
# 		
#         BMI: Body Mass Index
# 		
#         Diabetes Pedigree Function (DBF): A function that scores the likehood of diabetes based on family history
# 		
#         Age: individual Age (years)
# 		
#         Outcome: 0 = healthy individual, 1 = Diabetic individual
# This will involve the use of a supervised learning model (linear regression e.t.c). 
# 
# It is a regression task as our objective is to predict the outcome label ( 1 or 0).
# 
# Furthermore, it will use Batch Learning as it the model will be trained on all our current available data.

# # Getting the Data
# 
# we will be using a kaggle dataset that was madee available by National Institute of Diabetes 
# 
# and Digestive and Kidney Diseases” as part of the Pima Indians Diabetes Database.
# 
# steps to be done:
# 
#           (a) Convert the data to a format you can easily manipulate (ex: changing to all numerical)
# 		  (b) Check the size and type of data (time series, sample).
# 		  (C) Sample a test set, put it aside. 

# In[11]:


import pandas as pd 
import numpy as np 
diabetes = pd.read_csv('diabetes_1.csv')
#x includes every single column aside from Outcome (what we are trying to predict)
X = diabetes.drop('Outcome', axis = 1)
#y is our Outcome - either a 0 or 1 (diabetic or not diabetic)
y = diabetes["Outcome"]

#checking to see if every column in data frame is all numerical before moving on 
#returns True if all column is Numeric 
diabetes.shape[1] == diabetes.select_dtypes(include=np.number).shape[1]


# Data is all Numerical - so we can move to the next step

# # Explore the data: 
# 
# in this step we will try to get meaningful insight into our data any correlation between attributes 
# 
# steps to be done: 
# 
#           (a): Study each attribute and its characteristics ( this includes the Name, Type(int/ float, text, structured)
# 		  and identify correlation between attributes 
#           
# 		  (b): check the % of missing values in data
# 
# 		  (c): we Study how we would solve this problem manually.
# 		  (d): Identify extra data that would be useful (maybe in improving model performance)
# 
# 		  (e): we are going to outline the usefulness of the task/steps done above
# 		  (f): Document what you have learned from Data exploration

# In[9]:


#plot histogram for each numerical attribute  
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
diabetes.hist(bins=50, figsize=(20,15))
plt.show()


# In[48]:


#we are going to check for any missing values in our data 
if all(diabetes.isna().sum()) == 0:
    print('True')
else: 
    print('False')


# In[19]:


#this would be a case of multiple linear regression 
#y = b0 + b1x1 + b2x2 + ...+bnxn


# # DATA PREPARATION:
# 	In this step we get the data ready for machine learning 
#     
#     steps to be done: 
# 	
#     1. Data cleaning:
# 		Fix or remove outliers (optional).
# 		Fill in missing values (e.g., with zero, mean, median…) or drop their rows (or
# 		columns)
# 	2. Feature selection (optional):
# 		Drop the attributes that provide no useful information for the task.
# 
# 	3. Feature engineering, where appropriate:
# 		Discretize continuous features.
# 
# 		Decompose features (e.g., categorical, date/time, etc.).
# 		Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.).
# 		Aggregate features into promising new features.
# 
# 	4. Feature scaling:
# 		Standardize or normalize features.

# * No missing values in dataset
# * feature scaling taken care of in next section 
# 
# * We need to now work on corr matrix - see correlation between various attributes and also try attribute combination and because we have some tail heavy distribution we can try to transform them - this includes (Age, DiabetesPedigreeFunction and maybe pregnancies)

# # Shortlisting Promising Models and initializing train and test data
# 
# Steps to be done:
# 
# 	(1). splitting up out training and test data and applying various models
# 
# 	(2). Performance Measure
# 		For Perfomance measure it depends on the data - but we consider both
# 		(a): MAE (Mean absolute Error)
# 		(b): RMSE ( Root MEan square Error)
# 	(3). Analyze the most significant variables for each algorithm.
# 
# 	(4). Analyze the types of errors the models make and What data would a human have used to avoid these errors?
# 
# 	(5). Perform a quick round of feature selection and engineering.
# 
# 	(6). Perform one or two more quick iterations of the five previous steps.
# 
# 	(7). Shortlist the top three to five most promising models, preferring models that
# 	make different types of errors.

# In[118]:


from sklearn.pipeline import Pipeline

#modelling 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def evualate_preds(y_true, y_preds):
    '''
    This will be used to evaluate how good our model is-
    A measure of Accuracy score
     A measure of Precision score
      A measure of Recall
       A measure of f1
    '''
    accuracy = accuracy_score(y_true, y_preds)
    precision  = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2), 
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"precsision: {precision:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    return metric_dict

# Normalizing continuous variables

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range = (0,1))



#initializing our training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

model = LogisticRegression(max_iter = 1500)
# scaler.fit(X_train)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

#checking for patterns in our training data
model.fit(X_train, y_train)
#checking how good our model is 
model.score(X_test, y_test)

#

rs_y_preds = model.predict(X_test)

rs_metrics =  evualate_preds(y_test, rs_y_preds)



# In[ ]:





# In[ ]:





# # Saving the Model
# 

# In[122]:



import pickle 

#saving an existing model to file 
pickle.dump(model, open("model.pkl", "wb"))


# In[123]:


#load a saved model

loaded_pickle_model = pickle.load(open("model.pkl", "rb"))
# #makes some predictions with saved model 
# pickle_y_preds = loaded_pickle_model.predict(X_test)
# evualate_preds(y_test, pickle_y_preds)


# # FINE TUNING SYSTEM
# 	
# 

# (A): Trying various Parameters to see if model can be improved and best parameters possible
# 
#     1: RandomizedSearchCv
#     2: GridSearchCv

# # 1. Random Search

# In[61]:


#1: 
from scipy.stats import uniform as sp_random
from sklearn.model_selection import RandomizedSearchCV
grid = {"class_weight":  ['balanced'], 
        "solver": ["liblinear", "sag", "saga", "newton-cg"],
        "intercept_scaling": sp_random() 
        }


np.random.seed(42)

x = diabetes.drop('Outcome', axis = 1)
y = diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

clp = LogisticRegression(n_jobs = 1)

rs_clp = RandomizedSearchCV(estimator = clp, param_distributions = grid, n_iter = 10, #number of model to try
                            cv = 5, verbose = 2)

rs_clp.fit(X_train, y_train)




# In[68]:


print(rs_clp.best_score_)
print(rs_clp.best_params_)
#testing the best fit 
print(rs_clp.score(X_test, y_test))


# In[63]:


rs_y_preds = rs_clp.predict(X_test)

rs_metrics =  evualate_preds(y_test, rs_y_preds)


# # Grid search

# In[65]:


grid_2 = {"class_weight":  ['balanced'], 
        "solver": ["liblinear","newton-cg"],
        "intercept_scaling": [2.5, 5.8, 7.9],
        "C": [0.001,0.01,0.1,1,1.4, 3.5, 6.4,10,100,1000]}

from sklearn.model_selection import GridSearchCV

np.random.seed(42)


x = diabetes.drop('Outcome', axis = 1)
y = diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

clp = LogisticRegression(n_jobs = 1)

gs_clp = GridSearchCV(estimator = clp, param_grid = grid_2, #number of model to try
                            cv = 5, verbose = 2)

gs_clp.fit(X_train, y_train)


# In[71]:


print(gs_clp.best_score_)
print(gs_clp.best_params_)
#testing the best fit 
print(gs_clp.score(X_test, y_test))
gs_y_preds = gs_clp.predict(X_test)

gs_metrics =  evualate_preds(y_test, rs_y_preds)


# # PRESENT SOLUTION:
# 
# 	In this step we are going to document what we have done, describing what worked and what did not, and system limitations

# We implemented this project which focussed on predicting individuals or patients with diabetes based on  a variety of criteria - This includes: 
# 
# (a): Glucose Level
# (b):Blood Pressure
# (c): SkinThickness 
# (d): Insulin 
# (e): BMI
# (f): DiabetesPedigreeFunction
# (g): Age 
# 
# We explored the data to find any correlations in our data or see if any attribute combinations were possible to yield a higher correlation score than set attributes (such as Glucose Level e.t.c) 
# 
# However, after scaling, training and testing the model it did yielded a score of 74.68.  
# 
# based on some data analysis of our diabetes dataset using the Weka Explorer, I came to the conclusion that some features were not important and did not sufficiently contribute in providing an accurate outcome and could be dragging the model performance down. 
# 
# Furthermore, after removing the following features: pregnancy, SkinThickness, Insulin, and diabetesPedigreeFunction, and training and testing the model it resulted in the following results: 
# 
#     Accuracy: 82.47%
#     precision: 0.93
#     recall: 0.50
#     F1 score: 0.65
# 
# Hyperparameter tuning did not improve model performance - 
#   
# Therefore, based on this results i have come up with 2 conclusions:
# 
# (a): The removal of certain features result in a better model performance and accuracy score 
# 
# (b): System Limitations: There was not sufficient data and therefore, could not get full performance on the model. 
# 
