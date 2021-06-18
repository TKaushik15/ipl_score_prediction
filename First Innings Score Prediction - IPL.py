#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd


# In[43]:


cd Desktop/


# In[44]:


# Loading the dataset
df = pd.read_csv('ipl.csv')


# In[45]:


df.head()


# In[46]:


df.isna().sum()


# In[47]:


df.info()


# In[48]:


# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)


# In[49]:


# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']


# In[50]:


df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


# In[51]:


# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]


# In[52]:


df.head()


# In[53]:


# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[54]:


# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[55]:


# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[56]:


df.head()


# In[57]:


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[58]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[59]:


# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# In[60]:


# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:





# In[61]:


#Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[63]:


ridge =Ridge()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-1, 1, 5, 10, 20, 30, 35, 40]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv=25)
ridge_regressor.fit(X_train, y_train)


# In[64]:


ridge_regressor.best_params_


# In[65]:


ridge_regressor.best_score_


# In[66]:


prediction = ridge_regressor.predict(X_test)


# In[67]:


import seaborn as sns 
sns.distplot(y_test-prediction)


# In[68]:


from sklearn import metrics
import numpy as np
print("Mean sq error", metrics.mean_squared_error(y_test, prediction))
print("Mean absolute error", metrics.mean_absolute_error(y_test, prediction))
print("Root mean sq error", np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[ ]:





# In[69]:


#Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[70]:


lasso =Lasso()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-1, 1, 5, 10, 20, 30, 35, 40]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv=25)
ridge_regressor.fit(X_train, y_train)


# In[71]:


lasso_regressor.fit(X_train, y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[72]:


prediction = ridge_regressor.predict(X_test)


# In[73]:


import seaborn as sns
sns.displot(y_test-prediction)


# In[74]:


print("Mean sq error", metrics.mean_squared_error(y_test, prediction))
print("Mean absolute error", metrics.mean_absolute_error(y_test, prediction))
print("Root mean sq error", np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[75]:


#Both lasso and ridge reg are giving same values for error


# In[77]:


# Creating a pickle file for the classifier
import pickle
ilename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[ ]:




