
# coding: utf-8

# In[2]:

import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from hpelm import ELM


# In[3]:

matrices_item = sio.loadmat('folds_item_cold_start.mat')
matrices_user = sio.loadmat('folds_item_cold_start.mat')
test_ids = matrices_user['grp1'].T
train_ids = matrices_user['g1'].T
train_set = matrices_user['train_1']
test_set = matrices_user['test_1']
user_info=[]
f =open('u.user','r')
profession_names=[]
profession_id = {}
ages=[]
ages_id = range(0,90,5)
i = 0
for x in f.readlines():
    if x.split('|')[3] not in profession_names:
        profession_names.append(x.split('|')[3])
        profession_id[x.split('|')[3]]=i
        i+=1
    ages.append(int(x.split('|')[1]))
    user_info.append(x.split('|')[:4])


# In[4]:

print user_info[0]
print profession_id
print min(ages)
print ages_id


# In[ ]:




# In[18]:

user_id=0
user_meta={}
for x in user_info:
    meta_data=""
    age = int(x[1])
    sex = x[2]
    profession = x[3]
    temp_age=[0]*len(ages_id)
    if sex=='M':
        temp_age.append(0)
        temp_age.append(1)
    else:
        temp_age.append(1)
        temp_age.append(0)
    temp_prof = [0]*len(profession_id)
    temp_prof[profession_id[profession]]=1
    for i in range(0,len(profession_id)):
        temp_age.append(temp_prof[i])
    user_meta[user_id]=temp_age
    user_id+=1
    for i in range(0,len(ages_id)):
        if ages_id[i]>age:
            temp_age[i-1]=1
            break


# In[17]:

print len(user_meta[0])


# In[22]:

f=open('u.item','r')
item_meta = {}
for x in f.readlines():
    idx = int(x.split('|')[0])
    temp =[]
    for y in x.split('|')[0][5:]:
        temp.append(y)
    item_meta[idx]=temp


# In[7]:

print train_set.shape
print train_ids.shape


# In[14]:

train_ratings=[]
train_features=[]
for x in range(0,len(train_set)):
    temp=np.asarray(train_set[x])
    indices = np.where(temp>0)
    for z in indices:
        for y in z:
            rating = temp[y]-1
            rating_encoding = [0]*5
            rating_encoding[rating]=1
            train_ratings.append(rating_encoding)
            train_features.append(user_meta[x]+item_meta[y+1])


# In[15]:

test_ratings=[]
test_features=[]
for x in range(0,len(test_set)):
    temp=np.asarray(test_set[x])
    indices = np.where(temp>0)
    for z in indices:
        for y in z:
            rating = temp[y]-1
            rating_encoding = [0]*5
            rating_encoding[rating]=1
            test_ratings.append(rating_encoding)
            test_features.append(user_meta[x]+item_meta[y+1])


# In[11]:

print len(test_ratings)
print len(train_ratings)


# In[24]:

for x in range(10,110,10):
    elm = ELM(len(train_features[0]),len(train_ratings[0]))
    elm.add_neurons(x, "rbf_linf")
    elm.train(np.asarray(train_features),np.asarray(train_ratings))
    Preds = elm.predict(np.asarray(test_features))
    predicted_labels = np.argmax(Preds, axis=1)
    true_labels = np.argmax(test_ratings, axis=1)
    print mae(true_labels,predicted_labels)/4


# In[25]:

for x in range(10,110,10):
    elm = ELM(len(train_features[0]),len(train_ratings[0]))
    elm.add_neurons(x, "sigm")
    elm.train(np.asarray(train_features),np.asarray(train_ratings))
    Preds = elm.predict(np.asarray(test_features))
    predicted_labels = np.argmax(Preds, axis=1)
    true_labels = np.argmax(test_ratings, axis=1)
    print mae(true_labels,predicted_labels)/4


# In[26]:

for x in range(10,110,10):
    elm = ELM(len(train_features[0]),len(train_ratings[0]))
    elm.add_neurons(x, "tanh")
    elm.train(np.asarray(train_features),np.asarray(train_ratings))
    Preds = elm.predict(np.asarray(test_features))
    predicted_labels = np.argmax(Preds, axis=1)
    true_labels = np.argmax(test_ratings, axis=1)
    print mae(true_labels,predicted_labels)/4


# In[ ]:



