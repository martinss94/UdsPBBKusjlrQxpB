#!/usr/bin/env python
# coding: utf-8

# In[15]:


import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


data = pd.read_csv('E:\marketing.csv')


# In[18]:


df= data.copy()


# In[21]:


df.shape


# In[19]:


df.dtypes.value_counts()


# In[20]:


sns.heatmap(df.isnull())


# In[22]:


df['y']


# In[23]:


df['y'].value_counts()


# In[24]:


df['y'].value_counts(normalize=True)


# In[24]:


for col in df.select_dtypes('object'):
    print(f'{col :-<50} {df[col].unique()}')


# In[47]:


for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts()


# In[51]:


df[df['y'] == 'yes']


# In[52]:


df[df['y'] == 'no']


# In[53]:


positive_df=df[df['y'] == 'yes']


# In[54]:


negative_df=df[df['y'] == 'no']


# In[25]:


df= data.copy()


# In[56]:


df.head()


# In[58]:


#TRAIN TEST - NETTOYAGE -ENCODAGE


# In[10]:


from sklearn.model_selection import train_test_split


# In[26]:


trainset,testset = train_test_split(df,test_size = 0.2, random_state = 0) 


# In[27]:


trainset['y'].value_counts()


# In[63]:


testset['y'].value_counts()


# In[64]:


# ENCODAGE


# In[65]:


code ={'no' :0,
       'yes':1,
       'married' :1,
       'single' :0,
       'divorced' :2,
       'unknown' :0,
       'primary' :1,
       'secondary' :2,
       'tertiary' : 3,
       'cellular' :1,
       'telephone' :2,
       'jan':1,
       'feb' :2,
       'mar' :3,
       'ap' :4,
       'may' :5,
       'jun' :6,
       'jul' :7,
       'aug' :8,
       'oct' :10,
       'nov' :11,
       'dec' :12

       
    
}


# In[66]:


for col in df.select_dtypes('object'):
    df[col] = df[col].map(code)


# In[67]:


df


# In[30]:


def imputation(df):
    return df.dropna(axis=1)


# In[31]:


def encoding(df):
    code ={'no' :0,
       'yes':1,
       'married' :1,
       'single' :0,
       'divorced' :2,
       'unknown' :0,
       'primary' :1,
       'secondary' :2,
       'tertiary' : 3,
       'cellular' :1,
       'telephone' :2,
       'jan':1,
       'feb' :2,
       'mar' :3,
       'ap' :4,
       'may' :5,
       'jun' :6,
       'jul' :7,
       'aug' :8,
       'oct' :10,
       'nov' :11,
       'dec' :12
      
    
}
    for col in df.select_dtypes('object').columns:
        df.loc[:,col]= df[col].map(code)
    return df
    


# In[32]:


def preprocessing(df):
    df = encoding(df)
    df = imputation(df)
    x = df.drop('y',axis=1)
    y = df['y']
    return x,y


# In[34]:


X_train,Y_train = preprocessing(trainset)
X_test,Y_test = preprocessing(testset)


# In[ ]:


# MODELING


# In[59]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif


# In[67]:


model = DecisionTreeClassifier(random_state =0)


# In[76]:


# EVALUATION PROCESS


# In[61]:


from sklearn.metrics import f1_score,confusion_matrix,classification_report
from sklearn.model_selection import learning_curve


# In[73]:


def evaluation(model):
    
    model.fit(X_train,Y_train)
    ypred =model.predict(X_test)
    
    print(confusion_matrix(Y_test,ypred))
    print(classification_report(Y_test,ypred))
    print(ypred)
    
    N,train_score,val_score = learning_curve(model,X_train,Y_train,
                                             cv =5, scoring = 'f1',
                                             train_sizes= np.linspace(0.1,1,10))
    
    plt.figure(figsize =(12,8))
    plt.plot(N,train_score.mean(axis =1),label ='train score')
    plt.plot(N,val_score.mean(axis =1),label ='validation score')
    plt.legend()
    

    


# In[65]:


evaluation(model)


# In[ ]:





# In[ ]:




