# Applying Multivariate Regression to an imaginary example
# The example is data about cars (such as price, mileage, model, cylinders etc)
# The idea is to figure out the price of the car depending on factors like mileage or number of cylinders

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# First loading the car data into a dataframe
df = pd.read_excel('cars.xls')
#print(df.head())

# Using pandas to split the dataframe into the value to predict (price) and factors to use for prediction (mileage, cylinder and doors)
scale = StandardScaler()
X = df[['Mileage', 'Cylinder', 'Doors']]
y = df[['Price']]

# Normalizing the data
X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())
#print(X)

# Creating a simple ordinary least squares model
est = sm.OLS(y, X).fit()
print(est.summary())
# we see suprisingly mileage (x1) and doors (x3) have a low coeffient, but cylinders (x2) has a high coeffient
# Why do doors not matter? Most likely because some expensive sport cars have just 2 doors
# that means the price seems to depend on cylinders much more than on mileage or doors
# R-squared is also low (0.064), that is interesting

# We can see that the price depends on the number of cylinders also like this:
print(y.groupby(df.Cylinder).mean())





