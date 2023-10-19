import pymongo
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

import mongodemo

# Connect to MongoDB
#client = pymongo.MongoClient("mongodb://35.178.144.244:27017/productDB")
#db = client.productDB

# Access the collection and retrieve data
#collection = db.salesData
#documents = list(collection.find())
documents = mongodemo.documents

# Print the documents
#for document in documents:
#    print(document)

# Create pandas dataframe
salesData = pd.DataFrame(documents)
print(salesData.head())

#Eliminate all the columns of the type other than numbers
salesData = salesData.select_dtypes(include=['int64','float64'])

#Eliminate missing values
print('salesData contains %d missing values' %(salesData.isnull().sum().sum()))
salesData.dropna(inplace=True)

print("")

#Check presence of missing
print('After the missing elimination, there are %d missing in the data' %(salesData.isnull().sum().sum()))
print(salesData.head())

#Select the X and Y
Y = salesData['total_sales']
X = salesData[['price', 'percent_negative_reviews', 'customer_rating']]
print(X)
print(Y)

#Plot the data
X1 = X['price']
plt.figure(figsize=(10,8))
plt.scatter(X1, Y)
plt.title('Scatter plot of price vs sales')
plt.xlabel("Price")
plt.ylabel("Sales")
plt.show()

X2 = X['percent_negative_reviews']
plt.figure(figsize=(10,8))
plt.scatter(X2, Y)
plt.title('Scatter plot of negative_reviews vs sales')
plt.xlabel("percent_negative_reviews")
plt.ylabel("Sales")
plt.show()

X3 = X['customer_rating']
plt.figure(figsize=(10,8))
plt.scatter(X3, Y)
plt.title('Scatter plot of customer rating vs sales')
plt.xlabel("customer_rating")
plt.ylabel("Sales")
plt.show()

#Split the data into training and test
df_training, df_test = train_test_split(salesData, train_size=0.7, test_size=0.3)
print('Length of training data:',len(df_training))
print('Length of test data:',len(df_test))
print("")

#Define X and Y for the univariate regression
price_training = df_training['price']
sales_training = df_training['total_sales']

#perform regression to estimate parameters
lr = LinearRegression().fit(price_training.values.reshape(-1,1), sales_training.values.reshape(-1,1))
reg = ["intercept", "price"]
coef = pd.DataFrame([lr.intercept_,lr.coef_[0]], reg, columns= ['coefficients'])
print(coef)

#Calculate R-squared and MSE
y_pred = lr.predict(price_training.values.reshape(-1,1))
rSquare_training_uni = r2_score(sales_training.values.reshape(-1,1), y_pred)
mse_error_training_uni = mean_squared_error(sales_training,y_pred)
print("R-squared (Training Data): ", rSquare_training_uni)
print("Mean Squared Error (Training Data): ", np.sqrt(mse_error_training_uni))

#Display fitted line on top of scatter plots
plt.figure(figsize=(10,8))
plt.scatter(price_training, sales_training)
plt.plot(price_training, y_pred, 'r')
plt.title('Best linear regression')
plt.xlabel("Price")
plt.ylabel("Sales")
plt.show()

#Compute mean squared error and the R-squared of the univariate linear model using the test data
price_test = df_test['price']
sales_test = df_test['total_sales']

y_pred_hat = lr.predict(price_test.values.reshape(-1,1))

rSquare_test_uni = r2_score(sales_test.values.reshape(-1,1), y_pred_hat)
mse_error_test_uni = mean_squared_error(sales_test,y_pred_hat)
print("R-squared (Test Data): ", rSquare_test_uni)
print("Mean Squared Error (Test Data): ", np.sqrt(mse_error_test_uni))
print("")

#MULTI-VARIATE regression
sales_training = df_training['total_sales']
predictors_training = df_training[['price', 'percent_negative_reviews', 'customer_rating']]

#perform regression to estimate parameters
lr_mul = LinearRegression().fit(predictors_training, sales_training.values.reshape(-1,1))
reg2 = ["intercept", 'price', 'percent_negative_reviews', 'customer_rating']
coef2 = pd.DataFrame(np.transpose(np.hstack(([lr_mul.intercept_],lr_mul.coef_))),index= reg2 , columns= ['coefficients'])
print(coef2)

#Calculate R-squared and MSE
y_pred_training_mul = lr_mul.predict(predictors_training)
rSquare_training_mul = r2_score(sales_training, y_pred_training_mul)
mse_error_training_mul = mean_squared_error(sales_training,y_pred_training_mul)
print("R-squared (Multi-variate Training Data): ", rSquare_training_mul)
print("Mean Squared Error (Multi-variate Training Data): ", np.sqrt(mse_error_training_mul))


sales_test = df_test['total_sales']
predictors_test = df_test[['price', 'percent_negative_reviews', 'customer_rating']]

#Calculate R-squared and MSE
y_pred_test_mul = lr_mul.predict(predictors_test)
rSquare_test_mul = r2_score(sales_test, y_pred_test_mul)
mse_error_test_mul = mean_squared_error(sales_test,y_pred_test_mul)
print("R-squared (Multi-variate Test Data): ", rSquare_test_mul)
print("Mean Squared Error (Multi-variate Test Data): ", np.sqrt(mse_error_test_mul))

print("")

print("Univariate R2 :", rSquare_test_uni)
print("Multivariate R2 :", rSquare_test_mul)
print('Improvement', round((rSquare_test_mul - rSquare_test_uni)/rSquare_test_uni,2))

print("Univariate MSE :", mse_error_test_uni)
print("Multivariate MSE :", mse_error_test_mul)
print('Improvement', round((np.sqrt(mse_error_test_mul) - np.sqrt(mse_error_test_uni))/np.sqrt(mse_error_test_uni),2))



#Polynomial regression
pr_mul = PolynomialFeatures(degree=2).fit_transform(predictors_training)

model = LinearRegression()
model.fit(pr_mul, sales_training)

#Calculate R-squared and MSE
y_pr_training = model.predict(pr_mul)
rSquare_pr_training = r2_score(sales_training, y_pr_training)
mse_error_pr_training = mean_squared_error(sales_training,y_pr_training)
print("R-squared (Polynomial Training Data): ", rSquare_pr_training)
print("Mean Squared Error (Polynomial Training Data): ", np.sqrt(mse_error_pr_training))


