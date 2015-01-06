
# coding: utf-8

# In[1]:

import random

def chunker(file, size):
    line_no = 0
    data = []
    for line in file:
            line = line.rstrip('\n')
            if line_no > size:
                yield data
            data.append(line.split('\t'))
            line_no += 1

            
def reservoir_sampler(file, reservior_size, max_iter=-1):
    reservoir = []
    cnt = 0
    for line in file:
        if cnt < reservior_size:
            reservoir.append(line.rstrip('\n').split('\t'))
            cnt += 1
            continue
        pick = random.randint(0, cnt)
        if pick <= reservior_size-1:
            reservoir[pick] = line.rstrip('\n').split('\t')
        cnt += 1
        print cnt
        if max_iter != -1 and cnt > max_iter:
            return reservoir
    return reservoir


# In[2]:


#file = open("/Users/thebanstanley/Documents/kaggle/criteo/dac/train.txt")
#data = chunker(file, 100).next()
#data = reservoir_sampler(file, 100, 200)


# In[2]:

import pandas as pd


# In[3]:

df = pd.io.parsers.read_csv("/Users/thebanstanley/Documents/kaggle/"
                            "criteo/output.csv", header=None)


# In[4]:

#df.describe()
#df.dtypes
df.columns
df[0].dtype == 'object'

# discussion on why we need to encode everything in scikit learn
#http://stats.stackexchange.com/questions/95212/improve-classification-with-many-categorical-variables


# In[5]:

#df[list(df[1:])].values
# one hot encoder seems a very bad idea when the cardinality
# of the category is large. So ditching this effort
# the onehotencoder seems having problem with nan, so I
# am re-coding it to -99999
"""
for i in df.columns:
    if df[i].dtype == 'object':
        df[i].fillna('nan', inplace=True)
    else:
        df[i].fillna(-99999, inplace=True)


label = df[0].values
feature = df[df.columns[1:]].values
from sklearn.preprocessing import OneHotEncoder
%debug
enc = OneHotEncoder()
enc.fit(feature)
feature
"""


# In[51]:

from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import scipy

# lets cast nan to be "missing" so the feature hasher wont complain
# abt it
for i in df.columns:
    if df[i].dtype == "object":
        df[i].fillna("missing", inplace=True)
    else:
        # later on scikitlearn complains abt nans
        df[i].fillna(-999999, inplace=True)

label = df[0].values
#features = df[df.columns[1:]].values
#print type(features)
hasher = FeatureHasher(n_features=1, input_type='string')

list_of_nd = []
for i in df.columns[1:]:
    print i
    if df[i].dtype == 'object':
        list_of_nd.append(hasher.fit_transform(df[i]))
    else:
        list_of_nd.append(scipy.sparse.coo_matrix(df[i].values).T)


# In[52]:

#df[33] = hasher.transform(df[33])
#print list_of_nd

#for i in list_of_nd:
    #print i.shape
print list_of_nd[35].dtype, df[1].values.dtype
print list_of_nd[35].shape, scipy.sparse.coo_matrix(df[1].values).shape
import numpy as np
import scipy

# looks like I cannot mix scipy sparse matrixes and numpy arrays. Convert
# everything into scipy and then work off of it.

features = scipy.sparse.hstack(list_of_nd)

print list_of_nd[0], df[0].values


# In[53]:

features


# In[54]:

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features.toarray(), label)


# In[58]:

classification = clf.predict(features.toarray())


# In[67]:

np.sum(classification == label)*100.0/1000000


# In[69]:

clf.score(features.toarray(), label)


# In[69]:




# In[ ]:



