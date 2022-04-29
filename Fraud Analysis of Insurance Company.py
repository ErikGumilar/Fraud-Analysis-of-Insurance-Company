#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Did you know that insurance claim **Fraud Cases** in **Indonesia** are still high? Especially the fraud case experienced by **BPJS Kesehatan Indonesia**.
# 
# ![](https://c.tenor.com/8wri-Dez2ZYAAAAC/what-meme.gif)
# 
# Yes, that is true. Based on a news release by CNBC Indonesia in October 2021 with the news title "[Dirut BPJS Kesehatan: SPI Rumah Sakit Diharapkan Cegah Fraud)](https://www.cnbcindonesia.com/news/20211010213224-4-282832/dirut-bpjs-kesehatan-spi-rumah-sakit-diharapkan-cegah-fraud)", Ali Ghufron as President Director of BPJS Kesehatan Indonesia said, "We need to understand together, all acts of fraud and deceitfulness, no matter how small in the implementation of the **JKN-KIS Program**, is an act that violates the law. Therefore, let us strengthen our synergy and commitment in realizing the management of the **JKN-KIS Program** that is free from all acts of fraud. The presence of a competent and certified SPI hospital is expected to optimally prevent fraud." He asked all health agencies to help prevent such fraudulent acts. Ali Ghufron also emphasized that apart from being able to proactively take steps to prevent fraud, detect indications of fraud. and foster an anti-fraud culture, the role of SPI RS can also encourage and ensure that services meet the quality standards set so as to increase participant satisfaction and patient safety.
# 
# ![](https://c.tenor.com/LzSZyFMZ25cAAAAC/steve-harvey-unbelievable.gif)
# 
# About 2 years before the news from CNBC Indonesia was released, around 2019, Tirto.id released a news story with the title, "[Hantu Fraud yang Mengancam Perusahaan Asuransi di Indonesia](https://tirto.id/hantu-fraud-yang-mengancam-perusahaan-asuransi-di-indonesia-d9lf)", stating that this fraudulent act in insurance claims occurred in at least three types of insurance at general insurance companies in Indonesia. . The first is travel insurance. Second, motor vehicle insurance. The last is shipping insurance (marine insurance). The modes of cheating of these three types of insurance vary. There are two modes of cheating in travel insurance. First, claims that claim to have lost or stolen luxury goods purchased abroad. In fact, the goods purchased are counterfeit goods. The second mode is by inflating medical funds abroad.

# # Business Problem
# 
# Now imagine if you are one of the Data Scientist Team who works in a well-known management consulting company and has a client from an insurance company, let's call it `"AOA"`, which conveys that in the company there are many cases of fraud committed by its customers.
# 
# ![](https://media0.giphy.com/media/l46Cy1rHbQ92uuLXa/giphy.gif?cid=a267dfa3fo65wzqe3s9crna6qaxhja8mwphytvn8ovo2wf5x&rid=giphy.gif&ct=g&1648512000057)
# 
# Your leader asks your team to conduct an analysis on `AOA` company customers who have many fraud cases to detect such fraud cases so that `AOA` companies will no longer experience this in the future. Your team immediately performs analysis on the data provided by the `AOA` company. The results of the analysis carried out will be submitted back to the `AOA` company to decide its their own course of action.

# # Fraud Detection on Insurance Company with Explanatory Model Analysis - Agnostic Model
# 
# 
# [Alexander S. Gillis](https://www.techtarget.com/contributor/Alexander-S-Gillis) on [Tech Target](https://towardsdatascience.com/how-to-deal-with-unbalanced-data-d1d5bad79e72) said, **Fraud Detection** is a set of activities undertaken to prevent money or property from being obtained through false pretenses.
# 
# Fraud detection is applied to many industries such as banking or insurance. In banking, fraud may include forging checks or using stolen credit cards. Other forms of fraud may involve exaggerating losses or causing an accident with the sole intent for the payout. Fraud is typically involves multiple repeated methods, making searching for patterns a general focus for fraud detection. For example, data analysts can prevent insurance fraud by making algorithms to detect patterns and anomalies.

# ## Install Packages & Import Packages

# In[1]:


get_ipython().system('pip install dalex')
get_ipython().system('pip install scikit-plot')
get_ipython().system('pip install shap')
get_ipython().system('pip install eli5')
get_ipython().system('pip install lime')


# In[4]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install xgboost')


# In[5]:


# import pandas for data wrangling
import pandas as pd
# import numpy for vectorize data manipulation
import numpy as np
# import matplotlib.pyplot module for data visualization
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.pylab import rcParams
# import seaborn for data visualization
import seaborn as sns
# import scipy for certain statistical function
from scipy import stats
# features scaling
from sklearn.preprocessing import MinMaxScaler

# import train and test split method from scikit-learn
from sklearn.model_selection import train_test_split
# import metrics method for model evaluation
import sklearn.metrics as metrics
# import logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# import knn classifier
from sklearn.neighbors import KNeighborsClassifier
# import random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
# import xgboost classifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, datasets
from scipy.stats import uniform
from sklearn.pipeline import Pipeline

# load scikit-plot modules
import scikitplot as skplt

# import dalex to explain complex model
import dalex as dx

# load shap package for shap explanation
import shap
pd.set_option("display.max_columns",None)
shap.initjs()

# load eli5
import eli5

# load LimeTabularExplainer for LIME method
from lime.lime_tabular import LimeTabularExplainer 


# ## Load Dataset

# In[6]:


missing_values = ['?', '--', ' ', 'NA', 'N/A', '-'] #Sometimes Missing Values are't in form of NaN
data = pd.read_csv("https://raw.githubusercontent.com/hadimaster65555/dataset_for_teaching/main/dataset/car_insurance_fraud_dataset/insuranceFraud.csv", sep= ',', na_values = missing_values)


# In[8]:


data.head()


# ## Data Descriptions

# In[9]:


# data understanding and profilling
print("*"*8,"Data Understanding & Profilling","*"*8)
print("\n")

# data shape
print("There is", data.shape[0], "observation and", data.shape[1], "columns in this dataset")
print("\n")

# Data Information 
print("Data Info:", data.info())
print("\n")

# Numerical and Categorical Column
numerical= data.select_dtypes('number').columns
categorical = data.select_dtypes('object').columns

print(f'Numerical Columns:  {data[numerical].columns}')
print('\n')
print(f'Categorical Columns: {data[categorical].columns}')
print('\n')

# Statistical Summary of The Data
print("Statistical Description of Data:", data.describe())


# In this dataset there are 39 features. The details are as follows:
# 
# 
# 1.   `months_as_customer` = The length of time a user has been a customer (in months).
# 2.   `age` = Customer's age.
# 3.   `policy_number` = Customer's policy number.
# 4.   `policy_bind_date` = Customer's policy date.
# 5.   `policy_state` = Customer's policy state.
# 6.   `policy_csl` = Combined single limit polis.
# 7.   `policy_deductable` = Customer's joint risk.
# 8.   `policy_annual_premium` = Customer's annual premiums.
# 9.   `umbrella_limit` = Risk payment limit by the insurance company.
# 10.   `insured_zip` = Customer's ZIP Code.
# 11.   `insured_sex` = Customer's gender.
# 12.   `insured_education_level` = Customer's education level.
# 13.   `insured_occupation` = Customer's occupation.
# 14.   `insured_hobbies` = Customer's hobbies.
# 15.   `insured_relationship` = Customer's marital status.
# 16.   `capital-gains` = Large gains obtained from compensation.
# 17.   `capital-loss` = The amount of loss obtained from compensation.
# 18.   `incident_date` = the Customer's date the incident occurred.
# 19.   `incident_type` = Customer's incident type.
# 20.   `collision_type` = Customer's collision type.
# 21.   `incident_severity` = Severity.
# 22.   `authorities_contacted` = Contacted authorities.
# 23.   `incident_state` = Incident location, state.
# 24.   `incident_city` = The city where the incident occurred.
# 25.   `incident_location` = The location of the incident.
# 26.   `incident_hour_of_the_day` = The time the incident occurred.
# 27.   `number_of_vehicles_involved` = The number of vehicles involved in the incident.
# 28.   `property_damage` = Was there any property damage?
# 29.   `bodily_injuries` = Number of injured.
# 30.   `witness` = Number of witnesses
# 31.   `police_report_available`= Is there a police report?
# 32.   `total_claim_amount` = Customer's total claim.
# 33.   `injury_claim` = Claims for injuries.
# 34.   `property_claim` = Claims for property damage.
# 35.   `vehicle_claim` = Claims for car damage
# 36.   `auto_make` = Automotive brand.
# 37.   `auto_model` = Automotive model.
# 38.   `auto_year` = Year of car manufacture.
# 39.   `fraud_reported` = Was it reported as fraud? Y if yes.

# ## Feature Engineering
# 
# [Harshil Patel](https://harshilp.medium.com/) said on his article -[What is Feature Engineering — Importance, Tools and Techniques for Machine Learning](https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10)- Feature engineering is a machine learning technique that leverages data to create new variables that aren’t in the training set. It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy. Feature engineering is required when working with machine learning models. Regardless of the data or architecture, a terrible feature will have a direct impact on your model.

# ### Unique Value

# In[10]:


# unique value in dataset
for i in data.columns:
    print(F'{i}:',len(data[i].unique()))


# In[12]:


print(f'There are:',data['policy_state'].unique(),'unique value on customers policy state column')
print(f'and the count of those unique value are:','\n', data['policy_state'].value_counts())
print('\n')
print(f'There are:',data['policy_deductable'].unique(),'unique value on customers joint risk column')
print(f'and the count of those unique value are:','\n', data['policy_deductable'].value_counts())
print('\n')
print(f'There are:',data['insured_sex'].unique(),'unique value on customers gender column')
print(f'and the count of those unique value are:','\n', data['insured_sex'].value_counts())
print('\n')
print(f'There are:',data['insured_education_level'].unique(),'unique value on customers educational level column')
print(f'and the count of those unique value are:','\n', data['insured_education_level'].value_counts())
print('\n')
print(f'There are:',data['insured_occupation'].unique(),'unique value on customers occupation column')
print(f'and the count of those unique value are:','\n', data['insured_occupation'].value_counts())
print('\n')
print(f'There are:',data['insured_relationship'].unique(),'unique value on customers marital status column')
print(f'and the count of those unique value are:','\n', data['insured_relationship'].value_counts())
print('\n')
print(f'There are:',data['incident_type'].unique(),'unique value on customers incident type column')
print(f'and the count of those unique value are:','\n', data['incident_type'].value_counts())
print('\n')
print(f'There are:',data['fraud_reported'].unique(),'unique value on fraud report column')
print(f'and the count of those unique value are:','\n', data['fraud_reported'].value_counts())
print('\n')
print(f'There are:',data['police_report_available'].unique(),'unique value on police report column')
print(f'and the count of those unique value are:','\n', data['police_report_available'].value_counts())


# As you can see, there is columns that have many unique values with similar meanings, for example the level of education of the customer, such as, `JD`, `MD` and `PhD` have a doctoral level of education.
# 
# The same thing happens in the marital status column or the relationship status of the customer. `Unmarried` and `not-in-family` have the same meaning, such as `single`. `Husband` and `Wife` have the meaning that the customer is in a marriage bond. But then, there is a question. Why we do not change the `other-relative` and `own-child` not as `single` or other value? Because of we didn't know the true meanings of those two value.

# ### Change the Unique Value

# In[13]:


# Replace value of Education
data['insured_education_level'] = data['insured_education_level'].replace(["JD","MD","PhD"],"Doktoral")
data['insured_education_level'] = data['insured_education_level'].replace(["College"],"Bachelor")
print(f'Unique in Education level column became shortened into:', data['insured_education_level'].unique())
print('\n')

# Replace value of Marital Status
data['insured_relationship'] = data['insured_relationship'].replace(["not-in-family","unmarried"],"single")
data['insured_relationship'] = data['insured_relationship'].replace(["husband","wife"],"married")
print(f'Unique in Marital Status column became shortened into:', data['insured_relationship'].unique())
print('\n')


# ### Drop Columns
# 
# There are a lot of unnecessary columns on this dataset and we need to drop those columns.

# In[14]:


del data['policy_number']
del data['policy_bind_date']
del data['insured_zip']
del data['incident_location']
del data['incident_date']


# ***Explaination***
# 
# Why do we need to drop those columns?
# 
# 
# 
# 1.   `policy_number`
# 
#     When we wanna do a prediction due fraud case there aren't relation between policy number and fraud case. So we need to drop this column to minimize our Machine Learning.
# 
# 2.   `policy_bind_date`
# 
#     This column dropping have the same reason with `policy_number`. Because of there isn't relation between fraud case and customer's policy date.
# 
# 3.   `insured_zip`
# 
#     The insurance zip code there is only related to customer state location. So there isn't reason for us to do ML with this column.
# 
# 4.   `incident_location`
# 
#     The fraud case cannot be associated to incident location. Maybe some of us think that incident location related to criminal rate but there isn't relation between these 2 columns.
# 
# 5.   `incident_date`
# 
#     Have the same reason with `incident_location`. Of course some of us think that special date whose need a lot of money like Christmas can be a catalyst due fraud case but we do not discuss about that.

# ### Data Cleaning
# 
# Data cleaning is the process of fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data within a dataset. When combining multiple data sources, there are many opportunities for data to be duplicated or mislabeled.

# #### Missing Value

# In[15]:


# Missing value Check on all of variable
data.isnull().sum()


# There are 3 types of missing values:
# 
# 
# 
# 1.   Missing Completely at Random (MCAR)
# 
#     MCAR occurs when the missing on the variable is completely unsystematic. When our dataset is missing values completely at random, the probability of missing data is unrelated to any other variable and unrelated to the variable with missing values itself. For example, MCAR would occur when data is missing because the responses to a research survey about depression are lost in the mail.
# 
# 2.   Missing at Random (MAR)
# 
#     MAR occurs when the probability of the missing data on a variable is related to some other measured variable but unrelated to the variable with missing values itself. For example, the data values are missing because males are less likely to respond to a depression survey. In this case, the missing data is related to the gender of the respondents. However, the missing data is not related to the level of depression itself.
# 
# 3.   Missing Not at Random (MNAR)
# 
#     MNAR occurs when the missing values on a variable are related to the variable with the missing values itself. In this case, the data values are missing because the respondents failed to fill in the survey due to their level of depression.
# 
# 
# From the definition of type in Missing Value, in our dataset can be indicated as MNAR which is the missing values that occured because of Customer of Insurance Company must be depressed of the incident or their lying.

# Label Encoding for the fraud reported column.

# In[16]:


# index report on fraud case as 1 for Y, 0 for otherwise
data['fraud_reported'] = data['fraud_reported'].replace({'Y': 1,'N': 0})


# **Fill the Missing Value**
# 
# As you know, we has 3 columns that contain with missing value in it.
# Those columns are:
# 
# 
# 1.   `collision_type`
# 
#       For Collision type we can fill it with Mode of `collision_type` column.
# 
# 2.   `property_damage`
# 
#       For report on property damage we can fill it with Mode of `property_damage` column.
# 
# 3.   `police_report_available`
# 
#       For avaibility report on police officer, we can fill it with condition. If in authorities_contacted = 'Police', fill the NaN with YES, and NO in otherwise.

# In[17]:


# replace missing values of collision_type column
data['collision_type'] = data['collision_type'].fillna(data['collision_type'].mode()[0])

# replace missing values of property_damage column
data['property_damage'] = data['property_damage'].fillna(data['property_damage'].mode()[0])

# # replace missing values of police_report_available column
missing_mask = data['police_report_available'].isna()
mapping_dict = dict({'Police': 'YES' , 'Fire': 'YES', 'Ambulance': 'YES'})
data.loc[missing_mask, 'police_report_available'] = data.loc[missing_mask, 'authorities_contacted'].map(mapping_dict)
data['police_report_available'] = data['police_report_available'].fillna('NO')


# In[18]:


# recheck the Null
data.isnull().sum()


# #### Duplicated Values

# In[19]:


# checking for duplicated rows
data.duplicated().sum()


# ## **Exploratory Data Analysis**
# 
# [Prasad Patil](https://medium.com/@theprasadpatil) released an article entitled "[What is Exploratory Data Analysis?](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)" on the [Toward Data Science](https://towardsdatascience.com/) page, Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.

# ### **Distribution Visualizations of Features**

# In[20]:


# fraud reported percentage
label = 'Yes','No'
plt.pie([len(data['fraud_reported'][data['fraud_reported']==1]),len(data['fraud_reported'][data['fraud_reported']==0])],labels=label,autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Was it reported as fraud?')


# In[21]:


# customer relationship percentage
label = 'Married','Single','Own Child','Other Relation'
plt.pie([len(data['insured_relationship'][data['insured_relationship']=='married']),len(data['insured_relationship'][data['insured_relationship']=='single']),len(data['insured_relationship'][data['insured_relationship']=='own-child']),len(data['insured_relationship'][data['insured_relationship']=='other-relative'])],labels=label,autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Customers Relationship Status')


# In[22]:


# gender percentage
label = 'Male','Female'
plt.pie([len(data['insured_sex'][data['insured_sex']=='MALE']),len(data['insured_sex'][data['insured_sex']=='FEMALE'])],labels=label,autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Customers Gender')


# In[23]:


# report on authority percentage
label = 'Yes','No'
plt.pie([len(data['police_report_available'][data['police_report_available']=='YES']),len(data['police_report_available'][data['police_report_available']=='NO'])],labels=label,autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Police Report Available?')


# ### **Outliers Visualization**

# In[24]:


outlier = data[['months_as_customer','age','total_claim_amount','injury_claim','property_claim','vehicle_claim','capital-gains','capital-loss']]

fig = px.box(outlier.melt(), y="value", facet_col="variable",facet_col_wrap=2, boxmode="overlay", color="variable",height=1000, width=900)
fig.update_yaxes(matches=None)

for i in range(len(fig["data"])):
    yaxis_name = 'yaxis' if i == 0 else f'yaxis{i + 1}'
    fig.layout[yaxis_name].showticklabels = True

fig.update_layout(showlegend=False)
fig.update_xaxes(showline=True, linewidth=2, linecolor='grey')
fig.update_yaxes(showline=True, linewidth=2, linecolor='grey')

fig.show()


# ## **Data Preprocessing**

# ### **Label Encoding**

# In[25]:


# create new features as Vehicle Age
data['vehicle_age'] = 2022 - data['auto_year'] # Deriving the age of the vehicle based on the year value 
data['vehicle_age'].head(10)


# In[26]:


# drop the auto_year
del data['auto_year']


# In[27]:


data._get_numeric_data().columns


# In[28]:


data.select_dtypes(include=['object']).columns


# In[29]:


dummies = pd.get_dummies(data[[
    'policy_state', 'policy_csl', 'insured_sex', 'insured_education_level',
       'insured_occupation', 'insured_hobbies', 'insured_relationship',
       'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_state', 'incident_city',
       'property_damage', 'police_report_available', 'auto_make',
       'auto_model']])


# In[30]:


df = pd.concat([dummies, data._get_numeric_data()], axis=1)  # joining numeric columns
df.head(2)


# ### **Split The Data**

# In[31]:


yVar = df['fraud_reported']
xVar = df.drop('fraud_reported', axis = 1)

# split data
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.25, random_state=30)


# ### **Normalization**
# 
# **Why do we need to scale the variables in our dataset?**
# 
# Some machine learning algorithms are sensitive to feature scaling while others are virtually invariant to it. Let me explain that in more detail.
# 
# [Aniruddha](https://www.analyticsvidhya.com/blog/author/aniruddha/) on her articles called "[Feature Scaling for Machine Learning: Understanding the Difference Between Normalization vs. Standardization](analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)", Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.

# In[32]:


col = df.drop('fraud_reported', axis= 1).columns
scaler = MinMaxScaler()
X_train[col] = scaler.fit_transform(X_train[col])
X_test[col] = scaler.fit_transform(X_test[col])


# ### **Function**

# In[33]:


def perform_randomized_search(features, target, model, hyperparams, kFolds):
  randomizedsearch = RandomizedSearchCV(model, hyperparams, cv = kFolds, verbose=1)
  best_model = randomizedsearch.fit(features, target)
  print("The mean accuracy of the model is:",best_model.score(features, target))
  print("The best parameters for the model are:",best_model.best_params_)


# In[34]:


def perform_gridsearch(features, target, model, hyperparams, kFolds):
  gridsearch = GridSearchCV(model, hyperparams, cv=kFolds, verbose=1)
  best_model = gridsearch.fit(features, target)
  print("The mean accuracy of the model is:",best_model.score(features, target))
  print("The best parameters for the model are:",best_model.best_params_)


# In[35]:


def execute_pipeline(features,target, model_list, kFolds):
  pipe = Pipeline([("classifier", RandomForestClassifier())])
  gridsearch = GridSearchCV(pipe, model_list, cv=kFolds, verbose=1, n_jobs=-1) # Fit grid search
  best_model = gridsearch.fit(features, target)
  print("The mean accuracy of the model is:",best_model.score(features, target))
  print("The best parameters for the model are:",best_model.best_params_)


# ## **Modeling**

# In[36]:


models = [
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','elasticnet'],
                 "classifier__C": [0.25,0.5,0.75,1.0],
                 "classifier__solver": ['lbfgs','newton-cg','liblinear','sag','saga'],
                 "classifier__fit_intercept": [True,False],
                 "classifier__max_iter": [100, 1000,2500, 5000, 10000]
                 },
                {"classifier": [KNeighborsClassifier()],
                 "classifier__leaf_size": list(range(1,5)),
                 "classifier__n_neighbors": list(range(1,5)),
                 "classifier__p":[1,2]
                 },
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 50, 100, 1000],
                 "classifier__max_depth":[3,5,10,None],
                 "classifier__criterion": ['gini','entropy'],
                 "classifier__max_features": ['sqrt','log2'],
                 },
                {"classifier": [XGBClassifier()],
                 "classifier__n_estimators":[10, 50, 100],
                 "classifier__max_depth":[3,5,10,None],
                 "classifier__learning_rate":[0.1, 0.05, 0.01],
                 "classifier__grow_policy": [0,1]},]


# ### **Base Line**

# In[39]:


# create logistic regression baseline
logistic = LogisticRegression()
penalty = ['elasticnet', 'l2']
C = [0.25,0.5,0.75,1.0]
fit_intercept = [True,False]
hyperparameters = dict(C=C, penalty=penalty, fit_intercept=fit_intercept)


# In[40]:


# create KNN
knn = KNeighborsClassifier()

# hyperparameters that we want to tune.
leaf_size = list(range(1,5))
n_neighbors = list(range(1,5))
p=[1,2]

# convert to dictionary
hyperparameters_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)


# In[41]:


perform_randomized_search(X_train, y_train, logistic, hyperparameters, 10)


# In[42]:


perform_gridsearch(X_train, y_train, logistic, hyperparameters, 10)


# In[43]:


perform_randomized_search(X_train, y_train, knn, hyperparameters_knn, 10)


# In[44]:


perform_gridsearch(X_train, y_train, knn, hyperparameters_knn, 10)


# ### Execute PipeLine

# In[45]:


execute_pipeline(X_train, y_train, models, 5)


# Logistic Regression are the best model on our dataset.

# In[46]:


lg = LogisticRegression(penalty= 'l2', C = 0.75, solver = 'lbfgs', fit_intercept= True, max_iter= 100)
lg = lg.fit(X_train, y_train)

# # predict
y_pred_logreg = lg.predict(X_test)
y_pred_proba_logreg = lg.predict_proba(X_test)

# report result
logreg_report = pd.DataFrame(metrics.classification_report(y_test, y_pred_logreg, target_names=['Fraud','Not a Fraud'], output_dict=True))
print("Below is the report of Logistic Regression Classifier model results:","\n",logreg_report,'\n')


# In[47]:


logreg_cm = skplt.metrics.plot_confusion_matrix(y_test,y_pred_logreg)
print('Confusion Matrix on Logistic Regression Tuning:\n',logreg_cm)


# In[48]:


# model performance check
logreg_ROC_AUC = skplt.metrics.plot_roc_curve(y_test, y_pred_proba_logreg)
print(logreg_ROC_AUC)


# In[49]:


# check f1-score
logreg_f1_score = metrics.f1_score(y_test,y_pred_logreg)
print("The F1 score of Logistic Regression is:",logreg_f1_score,'\n')

# Test Score of logistic regression
logreg_accuracy = metrics.accuracy_score(y_test, y_pred_logreg)
print("The score for Logistic Regression is:",logreg_accuracy)


# ## **Model Agnostic**

# In[50]:


# initiate explainer for Logistic Regression model
fraud_lg_exp = dx.Explainer(lg, X_train, y_train, label = "Logistic Regression Interpretation")


# In[51]:


# visualize permutation feature importance for Logistic Regression model
fraud_lg_exp.model_parts().plot()


# ### **Partial Dependent Plot**

# In[52]:


# create partial dependence plot of Logistic Regression model
fraud_lg_exp.model_profile().plot()


# ## Shapley Value and Shapley Additive Explanations

# In[53]:


# create function for Logistic Regression model
f = lambda x: lg.predict_proba(x)[:,1]
# create median
med = X_train.median().values.reshape((1,X_train.shape[1]))

# create explainer for Logistic Regression model
lg_explainer = shap.Explainer(f, med)
# implement SHAP method for Logistic Regression model
lg_shap_values = lg_explainer(X_train[:1000])


# In[54]:


# create SHAP summary plot to visualize impact of next 1000 rows of train data
shap.summary_plot(lg_shap_values, X_train[1001:2000], plot_type='bar')


# In[55]:


# create SHAP summary plot to visualize impact distribution
shap.plots.beeswarm(lg_shap_values)


# ### Instance Level

# In[56]:


# initiate javascript module
shap.initjs()


# In[57]:


X_train.iloc[1,:]


# In[58]:


y_train[1]


# ### Lime Explainer

# In[61]:


# define Logistic Regression explainer with lime module
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names = X_train.columns.tolist(),
    class_names = ['Fraud', 'Not a Fraud'],
    discretize_continuous = True,
    verbose = True
)


# In[62]:


X = df.reset_index()
X

From analysis that already done, the biggest factors of Fraud report is Major Damage of Incident Severity.