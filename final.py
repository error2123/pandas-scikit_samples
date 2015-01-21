
# coding: utf-8

                The below procedure explains what to do when u have a dataset. But most times u have to creat one. I will be summarizing some of best practices when collecting and scrubbing data and getting it ready for analysis.

links: http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/

TODO
                
                Sample Template for Data analysis of a dataset:

- First Check if ur data set fits in memrory. IF too big use reservoir sampling to scope bring it down. I usually like abt 1GB.
You can do more than 1GB if you have more RAM. Try to hit a balance between more data and the speed at which u machine can analyize. Nothing spoils a enthusiastic Data scientist than idealing 30 minutes for something to show up on ur inotebook.

                
# In[3]:

# Read the data into a pandas frame.


get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'pylab inline')
import pandas as pd


# In[1]:

headers = ("id,click,hour,C1,banner_pos,site_id,"
        "site_domain,site_category,app_id,app_domain,"
        "app_category,device_id,device_ip,device_model,"
        "device_type,device_conn_type,C14,C15,C16,C17,C18,C19,"
        "C20,C21")
headers = headers.split(",")
df = pd.io.parsers.read_csv("/Users/thebanstanley/"
                            "Documents/kaggle/avazu/output.csv", header=None, 
                            delimiter=",", names=headers)


                Summarizing the data structure is about describing the number and data types of attributes. For example, going through this process highlights ideas for transforms in the Data Preparation step for converting attributes from one type to another (such as real to ordinal or ordinal to binary).

Some motivating questions for this step include:

How many attributes and instances are there?
What are the data types of each attribute (e.g. nominal, ordinal, integer, real, etc.)?

Definitions:
Nominal - categorical
Ordinal - Has order in the categories
Integer - ints
Real - floats
                
# In[2]:

df.describe
df.dtypes


                If you are working on ads space or ecommerce space, there is a good chance most of your features are categorical. This will be causing a lot of problems down the road when u r trying to plot data or train a regression on it. The simple way is to take all possible values for a categorical variable as u encounter them and then assign a number to it. There are a couple of problems with it, u have to read all ur data and essentially encode the categorical values into features. This is not feasible esp. when u are sampling from a big distribution or trying to parallelize and train multiple models. Also, when assigning ints to categorical variables you can unintentionally introduce some ordering in ur labels that ur model can try to hone into. 

A better way to do it is what is called the "hashing trick" which is pretty simply given a non-numeric key compute the numeric hash value for it(read more abt hash map data structure implementations to get a better understanding of how this is done and issues you need to be aware of like hash collisions).
                
# In[2]:

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


# In[ ]:

# pickle things esp. it takes a while to load the data and transform it.
df.to_pickle("/Users/thebanstanley/"
            "Documents/kaggle/avazu/df.pickled")
df.describe
df.dtypes


# In[4]:

# start here if u dont want to do all the preprocessing again

df = pd.io.pickle.read_pickle("/Users/thebanstanley/"
            "Documents/kaggle/avazu/df.pickled")


                Now, lets get a feel for data by plotting a boxplots. Things to look for:
- any outliers.
- how many data is missing

What we are looking for in histograms:
- feature-class relationship(only possible when u groupby the class and we plot it).

Lets plot box plots(mostly for real valued columns) but also makes some sense for categorical features we just hashed. Lets put histograms side by side.

Useful links:
http://machinelearningmastery.com/quick-and-dirty-data-analysis-for-your-machine-learning-problem/
http://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/

TODO:
Write more abt iterpretations of boxplots and histograms here.
                
# In[45]:

grouped = df.groupby('click')

for i in df.columns:
    
    # figure concept is very similiar to matlab, every time we do figure()
    # we get a new figure. You can label figures by saying figure(1). You
    # can show a figure using show() and the figure in that context will get
    # shown. If a new figure is created the older figure automatically seems
    # to get shown. More details here: http://matplotlib.org/users/pyplot_tutorial.html
    # esp. the section abt how figures work in matplotlib
    
    # 20 is the x axis width, 4 is the y axis width
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    plt.title("Boxplot for " + i)
    #fig.suptitle(i, fontsize=10)
    plt.boxplot(df[i])
    plt.subplot(122)
    # the groupby object in python is a little hard to work with.
    # Here we are iterating thru the groups. For every iteration we get
    # a tuple of (group_x, dataframe of data in group_x)
    for group in grouped:
        # reason for extracting the values is when I pass in a dataframe
        # the indexes dont start at zero(groupby preserves the indexes from
        # original dataframe) so hist errors out.
        plt.title("Histogram for " + i)
        plt.hist(group[1][i].values, label="click=" + str(group[0]))
        plt.legend()

# here is a way to do it if u just want to use dataframes histogram method
#grouped = df.groupby('click')
#
#for group in grouped:

#    print group[1]['device_type'].hist()


# In[ ]:

grouped = df.groupby('click')
# the df.columns seems to be an object, I am going to create
# a list out of it
list_of_indexes = {i for i in df.columns}
print "No. of features: ", len(list_of_indexes)
import itertools
marker = itertools.cycle(('o', '$', '+', '*', '^')) 

for col1 in df.columns:
    list_of_indexes.remove(col1)
    for col2 in list_of_indexes:
        plt.figure(figsize=(20,4))
        plt.title("X=" + col1 + " Y=" + col2)
        # this a way by which I can automatically choose colors.
        colors = iter(cm.rainbow(np.linspace(0, 1, len(grouped))))
        for group in grouped:
            # lets try to make markers look different too as if there
            # are overlaps in data it is still visible.
            plt.scatter(group[1][col1].values, group[1][col2].values, 
                        label="click=" + str(group[0]),
                        color=next(colors), marker=marker.next())
            plt.legend()


# takes too much time, so not using it
#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')



                Link for how to apply machine learning techniques:

When choosing machine learning algorithms go for something very simple. I start with a decision tree that uses information gain to make the splits so I can visualize the most discriminating features. How sparse the tree is etc.. Then I try something like a neural network as usually its predictiove power is high as it can do non-linear models but hard to explain.

- Also generative algorithm like a simple naive bayes can give insights into data that a discriminative technique may not.

Here is a exert from mlm:

Model Data
;;;;;;;;;;;;
Model accuracy is often the ultimate goal for a given data problem. This means that the most predictive model is the filter by which a model is chosen.

often the ‘best’ model is the most predictive model

Generally the goal is to use a model predict and interpret. Prediction can be evaluated quantitatively, whereas interpretation is softer and qualitative.

A model’s predictive accuracy can be evaluated by how well it performs on unseen data. It can be estimated using methods such as cross validation.

The algorithms that you try and your biases and reduction on the hypothesis space of possible models that can be constructed for the problem. Choose wisely.

For more information, take a look at How to Evaluate Models and How To Spot-Check Algorithms.

Interpret Results
;;;;;;;;;;;;;;;;;;;

The purpose of computing is insight, not numbers

— Richard Hamming

The authors use the example of handwritten digit recognition. They point out that a model for this problem does not have a theory of each number, rather it is a mechanism to discriminate between numbers.

This example highlights that the concerns of predicting may not be the same as model interpretation. In fact, they may conflict. A complex model may be highly predictive, but the number of terms or data transforms performed may make understanding why specific predictions are made in the context of the domain nearly impossible.

The predictive power of a model is determined by its ability to generalize. The authors suggest that the interpretative power of a model are its abilities to suggest the most interesting experiments to perform next. It gives insights into the problem and the domain.

The authors point to three key concerns when choosing a model to balance predictive and interpretability of a model:

Choose a good representation, the form of the data that you obtain, most data is messy.
Choose good features, the attributes of the data that you select to model
Choose a good hypothesis space, constrained by the models and data transforms you select.


http://machinelearningmastery.com/how-to-work-through-a-problem-like-a-data-scientist/
# http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/   
# http://machinelearningmastery.com/how-to-use-machine-learning-results/
                
# In[13]:

# converting label and feature to an ndarray
label = df['click'].values
# the decision tree classifier expect a shape of (no_of_instances,)
print shape(label)

df_shallow_copy = df.copy()
df_shallow_copy.drop('click', axis=1, inplace=True)
features = df_shallow_copy.values
# the decision tree classifier expect a shape of (no_of_instances, no_of_features)
print shape(features)


# In[14]:

# if u run into problems with nan values check this: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
#
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, label)


# In[17]:

from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data)


# In[ ]:

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf")


# In[ ]:



