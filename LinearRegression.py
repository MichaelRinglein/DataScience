# Applying Linear Regression to an imaginary example
# The example is the relation between page speed and amount purchased of an online shop
# The idea is to show that the faster the page loads, the more are people spending
# Or, in other words, displaying the correlation between page speed and amount purchased

# Interesting question: is more spend because the page loads faster or are wealthy people able to spend more for a fast internet connection?)
# This analysis can't answer this question. Good example that correlation often does not mean causation)
# Whatever the causation, this script visualizes the correlation anyway

import numpy as np
from pylab import *

# creating random data showing pagespeed and purchase amounts of an online shop in relation to eachother
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3
#plt.scatter(pageSpeeds, purchaseAmount)
#plt.show()

# getting slope, r_value etc.
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
print('Slope is:', slope)
print('R sqared is:', r_value ** 2)

# get a prediction using the slope
import matplotlib.pyplot as plt
def predict(x):
	return slope * x + intercept #y = mx + b
	
fitLine = predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()

