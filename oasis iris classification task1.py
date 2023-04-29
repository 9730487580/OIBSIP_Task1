#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('iris.csv')
df


# In[5]:


df.head()


# In[13]:


df.info()


# In[14]:


df.isnull().sum()


# In[17]:


df.columns


# In[23]:


df.describe()


# In[31]:


df['Species'].value_counts()


# In[35]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[34]:


x


# In[36]:


y


# In[37]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[38]:


x_train.shape


# In[39]:


x_test.shape


# In[40]:


y_train.shape


# In[41]:


y_test.shape


# In[42]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[43]:


model.fit(x_train,y_train)


# In[45]:


y_pred=model.predict(x_test)


# In[46]:


y_pred


# In[47]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[48]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[ ]:




