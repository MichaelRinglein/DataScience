import numpy as np
import matplotlib.pyplot as plt

# Simple outlier excercise
# Creating income distribution 
# 10,000 people, 27,000 mean income and standart deviation of 15,000
# Quite realistic

incomes = np.random.normal(27000, 15000, 10000)

# Lets add a billionaire with 10,000,000,000.00 (10 billions)
incomes = np.append(incomes, [10000000000])
plt.hist(incomes, 50)
plt.show() #We see that the billionaire completely 'messes up' the data

# How does the one billionaire influence the mean of the income?
print(incomes.mean()) #1,026,863.921 - a bit over one million is not the typical mean income

# Writing a function to detect and sort out outliers
# Everything of two standart deviations from the mean gets filtered out

def reject_outliers(data):
	u = np.median(data)
	s = np.std(data)
	filtered = [e for e in data if (u-2*s < e < u+2*s)] #2 standard deviations
	return filtered

filtered = reject_outliers(incomes) 

plt.hist(filtered, 50)
plt.show() #Now the distribution of incomes looks realistic again

print(np.mean(filtered)) #26,970.96 as mean income is quite close to our original 27,000 

