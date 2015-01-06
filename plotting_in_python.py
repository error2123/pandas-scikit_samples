
# coding: utf-8

# In[3]:

# LINE PLOT
#
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pylab as pl
# Make an array of x values
x = [1, 2, 3, 4, 5]
# Make an array of y values for each x value
y = [1, 4, 9, 16, 25]
# use pylab to plot x and y
pl.plot(x, y)
# show the plot on the screen
pl.show()


# In[25]:

# SCATTER PLOT
#

#character color
#b blue
#g green
#r red
#c cyan
#m magenta
#y yellow
#k black
#w white

#’s’ square marker
#’p’ pentagon marker
#’*’ star marker
#’h’ hexagon1 marker
#’H’ hexagon2 marker
#’+’ plus marker
#’x’ x marker
#’D’ diamond marker
#’d’ thin diamond marker


import numpy as np
import pylab as pl
# Make an array of x values
x = [1, 2, 3, 4, 5]
# Make an array of y values for each x value
y = [1, 4, 9, 16, 25]
# use pylab to plot x and y as red circles
pl.plot(x, y, 'g--')


# LABEL AXISES AND TITLES
pl.xlabel('x-axis')
pl.ylabel('y-axis')
# set axis limits
pl.xlim(0.0, 7.0)
pl.ylim(0.0, 30.)
# setting ranges
print pl.yticks(np.arange(0, 31, 7))

# show the plot on the screen
pl.show()


# In[27]:

# PLOTTING MORE THATN ONE PLOT ON THE SAME AXIS
x1 = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
x2 = [1, 2, 4, 6, 8]
y2 = [2, 4, 8, 12, 16]
# use pylab to plot x and y
pl.plot(x1, y1, 'r')
pl.plot(x2, y2, 'g')
# give plot a title
pl.title('Plot of y vs. x')
# make axis labels
pl.xlabel('x axis')
pl.ylabel('y axis')
# set axis limits
pl.xlim(0.0, 9.0)
pl.ylim(0.0, 30.)
# show the plot on the screen
pl.show()


# In[30]:

# PLOTTING LEGENDS
#http://matplotlib.org/users/legend_guide.html

# I NEED TO LOOK MORE INTO LEGENDS..

# Make x, y arrays for each graph
x1 = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
x2 = [1, 2, 4, 6, 8]
y2 = [2, 4, 8, 12, 16]
# use pylab to plot x and y : Give your plots names
plot1 = pl.plot(x1, y1, 'r')
plot2 = pl.plot(x2, y2, 'go')
# give plot a title
pl.title('Plot of y vs. x')
# make axis labels
pl.xlabel('x axis')
pl.ylabel('y axis')
# set axis limits
pl.xlim(0.0, 9.0)
pl.ylim(0.0, 30.)
# make legend
pl.legend([plot1, plot2], ('red line', 'green circles'), 'best', numpoints=1)
# show the plot on the screen
pl.show()


# In[32]:

# HISTOGRAMS
# make an array of random numbers with a gaussian distribution with
# mean = 5.0
# rms = 3.0
# number of points = 1000
data = np.random.normal(5.0, 3.0, 1000)
# make a histogram of the data array
pl.hist(data)
# make plot labels
pl.xlabel('data')
pl.show()


# In[33]:

bins = np.arange(-5., 16., 1.)
pl.hist(data, bins, histtype='stepfilled')


# In[35]:

f1 = pl.figure(1)
pl.subplot(221)
pl.subplot(222)
pl.subplot(212)
pl.subplots_adjust(left=0.08, right=0.95, wspace=0.25, hspace=0.45)


# In[ ]:



