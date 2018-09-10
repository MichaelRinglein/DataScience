# Applying Linear Regression to an imaginary example
# The example is the relation between page speed and amount purchased of an online shop
# The idea is to show that the faster the page loads, the more are people spending
# Or, in other words, displaying the correlation between page speed and amount purchased

# Interesting question: is more spend because the page loads faster or are wealthy people able to spend more for a fast internet connection?)
# This analysis can't answer this question. Good example that correlation often does not mean causation)
# Whatever the causation, this script visualizes the correlation anyway


from pylab import *
import numpy as np
import matplotlib.pyplot as plt

# creating random data showing pagespeed and purchase amounts of an online shop in relation to eachother
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
#plt.scatter (pageSpeeds, purchaseAmount)
#plt.show()

# creating a 4th degree polynominal model using numpy's polyfit function
x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 4)) #creates y = ax^4 + bx^3 + cx^2 + dx + e
xp = np.linspace(0, 7, 100) #from 0 to 7 secons of page speed
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show() #not sure if it isn't overfitting in the end after 6 seconds page load, but it looks reasonable from 0 to 6 seconds page load

# getting r_value etc.
from sklearn.metrics import r2_score
r2 = r2_score(y, p4(x))
print('r^2 of our 4th degree polynominal model is: ', r2)

# to test, let's create a 10th degree polynominal model and see if the data fits better or worse (-> overfitting)
p10 = np.poly1d(np.polyfit(x, y, 10)) #creates y = ax^10 + bx^9 + ....)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r') #4th dimensional again in red
plt.plot(xp, p10(xp), c='g') #10th dimensional in green
plt.show() #as we see, it doesn't really fit better

# getting r_value of the 10th degree polynominal model
r2_of10thdegree = r2_score(y, p10(x))
print('r^2 of our 10th degree polynominal model is: ', r2_of10thdegree)
 