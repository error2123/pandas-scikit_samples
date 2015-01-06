
# coding: utf-8

# In[74]:

import pandas as pd
get_ipython().magic(u'matplotlib inline')


# In[75]:

header = ['click', 'imp', 'url', 'adid', 'advertiserid', 'depth', 'position', 'queryid', 'keywordid', 'titleid', 
          'descriptionid', 'userid']

df = pd.read_csv('/Users/thebanstanley/Documents/kaggle/ctr_prediction/track2/tmp/smaller_training.txt', sep='\t', 
                           header=0, names=header, dtype=float64)


# In[76]:


df
# this will give u all dtypes of the columns
df.dtypes


# In[77]:

df.describe()


# In[78]:

plt = df.boxplot(column='imp')


# In[122]:

# PLOT MATRIX. LIKE WEKA. SO WE CAN SEE THE CORRELATION.
# SCATTER PLOT OF EVERY POSSIBLE COMBINATION.
import pylab as pl
from itertools import combinations
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr_n = linear_model.LinearRegression(normalize=True)
#i = 0
for pairs in combinations(header, 2):
    # figure seems to create a new plot everytime
    pl.figure()
    pair_0 = df[pairs[0]].values
    pair_1 = df[pairs[1]].values
    print "Pearson Coefficient", np.corrcoef(pair_0, pair_1)
    
    # REASON FOR DOING THIS IS TO SHOW THE PEARSON CORRELATION IS NOT SAME AS
    # LINEAR REGRESSION WEIGHTS. THEY ARE CORRELATED. BUT NOT SAME. THEY SEEM TO
    # BE STRONGLY CORRELATED WHEN WE ARE USING ONLY ONE FEATURE FOR LINEAR REGRESSION
    # IN CASE OF MULTIPLE FEATURES THEY MIGHT BE WEAKLY CORRELATED DEPENDING ON WHICH
    # FEATURE THE GRADIENT DESCENT ENDED UP RELYING ON MORE.
    # In the single predictor case of linear regression, 
    # the standardized slope has the same value as the correlation coefficient.
    
    # scikitlearn has this stupid error where we need to explicitly reshape
    # the ndarray to have x_samples and one feature dimension
    regr.fit(pair_0.reshape(pair_0.shape[0], 1), pair_1.reshape(pair_1.shape[0], 1))
    # The coefficients
    print('Coefficients no normalization: \n', regr.coef_, regr.intercept_)
    
    # NORMALIZATION DOES NOT SEEM TO WORK IN LINEAR REGRESSION SCIKIT. NEED TO FIGURE
    # THIS OUT
    regr_n.fit(pair_0.reshape(pair_0.shape[0],1), pair_1.reshape(pair_1.shape[0],1))
    
    # The coefficients
    print('Coefficients with normalization: \n', regr_n.coef_, regr_n.intercept_)
    
    pl.plot(pair_0, pair_1, "ro")
    pl.title("_".join(pairs))
    #i += 1
    #if i > 2:
    #    break




# In[82]:

print header
import copy
training_set = copy.deepcopy(header)
training_set.remove('click')
training_set.remove('imp')
print training_set
training_data = df[training_set].values
ctr = np.divide(df['click'].values.astype(np.float), df['imp'].values)
print [x for x in ctr if x > 0]



# In[93]:

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(training_data, ctr)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(training_data) - ctr) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(training_data, ctr))

import pylab as pl

# Plot outputs
pl.scatter(training_data[:,5], ctr,  color='black')
pl.plot(training_data, regr.predict(training_data), color='blue',
        linewidth=3)




# In[ ]:



