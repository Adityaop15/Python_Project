#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = r"https://drive.google.com/uc?export=download&id=13ZTYmL3E8S0nz-UKl4aaTZJaI3DVBGHM"
df  = pd.read_csv(path)


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


plt.scatter(x =df.study_hours, y = df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()


# In[9]:


#Prepare the data for Machine Learning algorithms


# In[10]:


df.isnull().sum()


# In[ ]:


# Data Cleaning


# In[13]:


df.mean()


# In[14]:


df2 = df.fillna(df.mean())


# In[15]:


df2.head()


# In[16]:


# split dataset


# In[17]:


X = df2.drop("student_marks", axis = "columns")
y = df2.drop("study_hours", axis = "columns")
print("shape of X = ", X.shape)
print("shape of y = ", y.shape)


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=51)
print("shape of X_train = ", X_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of X_test = ", X_test.shape)
print("shape of y_test = ", y_test.shape)


# In[19]:


# y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[20]:


lr.fit(X_train,y_train)


# In[21]:


lr.coef_


# In[22]:


lr.intercept_


# In[23]:


m = 3.93
c = 50.44
y  = m * 4 + c 
y


# In[24]:


lr.predict([[4]])[0][0].round(2)


# In[25]:


y_pred  = lr.predict(X_test)
y_pred


# In[26]:


pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])


# In[27]:


lr.score(X_test,y_test)


# In[28]:


plt.scatter(X_train,y_train)


# In[29]:


plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = "r")


# In[30]:


import joblib
joblib.dump(lr, "student_mark_predictor.pkl")


# In[31]:


model = joblib.load("student_mark_predictor.pkl")


# In[32]:


model.predict([[5]])[0][0]


# In[ ]:




