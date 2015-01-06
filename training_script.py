
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'pylab inline')
import pandas as pd


headers = ("id,click,hour,C1,banner_pos,site_id,"
        "site_domain,site_category,app_id,app_domain,"
        "app_category,device_id,device_ip,device_model,"
        "device_type,device_conn_type,C14,C15,C16,C17,C18,C19,"
        "C20,C21")
headers = headers.split(",")
df = pd.io.parsers.read_csv("/Users/thebanstanley/"
                            "Documents/kaggle/avazu/output.csv", header=None, 
                            delimiter=",", names=headers)


# In[2]:

#df.describe()
#df.dtypes
#for i in df.columns:
#    if df[i].dtype=="object":
#        df.fillna("missing", inplace=True)

# Pandas has problems plotting categorical values and with
# problems I was having with scikit learn too I decided to
# do the feature mapping as early as possible, so we get
# past all these errors with categorical variables
#
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=1, input_type='string')
for i in df.columns:
    if df[i].dtype=='object':
        df[i] =  hasher.fit_transform(df[i]).toarray()


# In[4]:

df.to_pickle("/Users/thebanstanley/"
            "Documents/kaggle/avazu/df.pickled")
df.dtypes


# In[13]:

#import matplotlib.pyplot as plt
#import pylab as pl
#%pylab inline
#pd.options.display.mpl_style = 'default'
#%matplotlib inline

# when plotting make sure u put all these lines on top of the notebook
# in that order
#
#%matplotlib inline
#import numpy as np
#import matplotlib.pyplot as plt
#%pylab inline
#import pandas as pd

#df.boxplot(column=None, fontsize=10, rot=90)

for i in df.columns:
    fig = plt.figure(figsize=(3,2))
    fig.suptitle(i, fontsize=10)
    plt.boxplot(df[i])


# In[12]:

#%debug
for i in df.columns:
    fig = plt.figure(figsize=(3,2))
    fig.suptitle(i, fontsize=10)
    plt.hist(df[i])


# In[20]:

#http://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
df.groupby('click').hist()


# In[ ]:

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')


# In[ ]:



