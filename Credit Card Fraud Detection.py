#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# In[4]:


credit_card_data=pd.read_csv('creditcard.csv')


# In[5]:


credit_card_data.head()


# In[6]:


credit_card_data.tail()


# In[7]:


credit_card_data.info()


# In[8]:


credit_card_data.isnull().sum()


# In[11]:


credit_card_data['Class'].value_counts()


# In[12]:


legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


# In[13]:


print(legit.shape)
print(fraud.shape)


# In[14]:


legit.Amount.describe()


# In[15]:


fraud.Amount.describe()


# In[16]:


credit_card_data.groupby('Class').mean()


# In[17]:


legit_sample=legit.sample(n=367)


# In[18]:


new_dataset=pd.concat([legit_sample,fraud],axis=0)


# In[19]:


new_dataset.head()


# In[20]:


new_dataset.tail()


# In[21]:


new_dataset['Class'].value_counts()


# In[22]:


new_dataset.groupby("Class").mean()


# In[23]:


x=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']


# In[24]:


print(y)


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[27]:


print(x.shape,x_train.shape,x_test.shape)


# In[28]:


model=LogisticRegression()


# In[29]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)


# In[30]:


model.fit(x_train,y_train)


# In[31]:


model.fit(x_train,y_train)


# In[32]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)


# In[33]:


model.fit(x_train,y_train)


# In[34]:


x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[35]:


print('Accuracy on training data:',training_data_accuracy)


# In[36]:


x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[37]:


print('Accuracy on Test data:',test_data_accuracy)

