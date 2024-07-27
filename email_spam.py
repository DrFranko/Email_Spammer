#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn
import numpy as np

import torch
from torch import nn


import re
import string
import nltk
stopwords = nltk.corpus.stopwords.words("english")
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



# In[95]:


df=pd.read_csv("C:\Datasets\spam_ham_dataset.csv\spam_ham_dataset.csv")
df.head()


# In[96]:


df.info()


# In[97]:


values=df['label'].value_counts()
values


# In[98]:


ps=PorterStemmer()
lm=WordNetLemmatizer()
def preprocess(text):
    #Punctuations
    text="".join([i for i in text if i not in string.punctuation])
    #Lowercase
    text=text.lower()
    #Tokenize
    text=re.split("W+",text)
    #Stopwords
    text=[i for i in text if i not in stopwords]
    #Stemmitiztaion
    text=[ps.stem(i) for i in text]
    #Lemmatization
    text=[lm.lemmatize(i) for i in text]
    #Joining the Tokens
    text=" ".join(text)
    return text


# In[99]:


df['text']=df['text'].apply(lambda x:preprocess(x) )
df.head()


# In[100]:


wc = WordCloud(width=500,height=500)
spam_wc = wc.generate(df[df['label_num'] == 1]['text'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)
plt.show()


# In[101]:


wc = WordCloud(width=500,height=500)
spam_wc = wc.generate(df[df['label_num'] == 0]['text'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)
plt.show()


# In[102]:


X=df['text']
y=df['label_num']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state = 2)


# In[103]:


vectorizer = CountVectorizer()
X_train_V = vectorizer.fit_transform(X_train)
X_test_V = vectorizer.transform(X_test)


# In[104]:


X_train_V.shape


# In[105]:


X_train_tensor = torch.tensor(X_train_V.toarray(),
                             dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values,
                             dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_V.toarray(),
                             dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,
                             dtype=torch.float32)


# In[106]:


from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset,batch_size=64)
test_loader = DataLoader(test_dataset,batch_size=64)


# In[107]:


class Classify(nn.Module):
    def __init__(self):
        super(Classify,self).__init__()
        self.Layer=nn.Sequential(
            nn.Linear(45470,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.Layer(x)


# In[108]:


model=Classify()


# In[109]:


criterion=nn.BCELoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)


# In[110]:


epochs=6
losses=0.0
for epoch in range(epochs):
    losses=0.0
    model.train()
    for text,label in train_loader:
        optimizer.zero_grad()
        y_pred=model(text)
        loss=criterion(y_pred,label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(f'Epoch {epoch+1}/{epochs} for Loss {losses}')


# In[111]:


model.eval()

correct = 0
total = 0
true_positives = 0
predicted_positives = 0
actual_positives = 0
with torch.inference_mode():
    for text, labels in test_loader:
        y_pred = model(text)
        predicted = (y_pred > 0.5).float()
        
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(dim=1)).sum().item()

        

        true_positives += ((predicted == 1) & (labels.unsqueeze(dim=1) == 1)).sum().item()
        predicted_positives += (predicted == 1).sum().item()
        actual_positives += (labels == 1).sum().item()

accuracy = correct / total
precision = true_positives / predicted_positives if predicted_positives > 0 else 0
recall = true_positives / actual_positives if actual_positives > 0 else 0
print('Test Accuarcy: {:.2f}%'.format(100 * accuracy))


# In[112]:


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.toarray())
            output = self.model(X_tensor)
            predictions = (output > 0.5).int().numpy().flatten()
        return predictions

wrapped_model = ModelWrapper(model)


# In[113]:


import pickle

with open('model.pkl','wb') as file:
    pickle.dump(wrapped_model,file)


# In[114]:


with open('vectorizer.pkl','wb') as file:
    pickle.dump(vectorizer,file)

