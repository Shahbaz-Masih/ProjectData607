# Explain the project file structure
# Importing the libraries
import numpy as np
import pandas as pd
# Pickle is used to serializing and de-serializing a Python object structure/data
import pickle

# copy and paste the data file onto the project folder, read the csv file using pandas, explore data and transform it
df = pd.read_csv('appData.csv')
print(df.head())
print(df.shape)
# check if there are any null or missing values and remove them
# print("Number of NaN values for the column temperature :", df['AT'].isnull().sum())
# print("Number of NaN values for the column exhaust_vacuum :", df['V'].isnull().sum())
# print("Number of NaN values for the column ambient_pressure :", df['AP'].isnull().sum())
# print("Number of NaN values for the column relative_humidity :", df['RH'].isnull().sum())
# print("Number of NaN values for the column energy_output :", df['PE'].isnull().sum())
# no null values found
# Use IQR score method to remove outliers
# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# # Use IQR score method to remove outliers
# df2 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(df2.shape)
# split data into training and testing data sets
df3 = df.copy()
df3.drop('Unnamed: 0', axis=1, inplace=True)
df3.columns = ['Age', 'Price', 'Property_Type', 'Duration', 'PPDCategory_Type']

df3_one_hot = df3.copy()
df3_one_hot = pd.get_dummies(df3_one_hot, columns=['Property_Type', 'Duration', 'PPDCategory_Type'])
import numpy as np
df3_one_hot['Age'] = np.where(df3_one_hot['Age'].str.contains('N'), 1, 0)

df3_one_hot.head()

from sklearn.model_selection import train_test_split

#data split with stratification
from sklearn.model_selection import train_test_split
df3_one_hot_model = df3_one_hot.copy()
y = df3_one_hot_model.pop('Age')
X = df3_one_hot_model
#data split with stratification
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state = 10)
# print("Size of X Train:\t",X_train.count())
# print("Size of X Test:\t",X_test.count())

# Import the linear regression model

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
classifier.fit(X_train, y_train)

# Saving model to disk by serializing the data objects writing it for using later on
pickle.dump(classifier, open('lrModel.pkl', 'wb'))

# Loading model to compare the results (by de-serializing and reading it)
lrModel = pickle.load(open('lrModel.pkl', 'rb'))
# print(lrModel.predict([[14, 50, 1000, 100]]))
# import decision tree regressor model
# from sklearn.tree import DecisionTreeRegressor
# dtregressor = DecisionTreeRegressor(criterion="mae")

# Fitting model with training data
# dtregressor.fit(X_train, y_train)
#
# # Saving model to disk
# pickle.dump(dtregressor, open('dtModel.pkl','wb'))
#
# # Loading model to compare the results
# dtModel = pickle.load(open('dtModel.pkl','rb'))
# print(dtModel.predict([[14, 50, 1000, 100]]))

# Import the linear regression model

# from sklearn.ensemble import RandomForestRegressor
# rfRegressor = RandomForestRegressor(criterion='mse')
#
# # Fitting model with training data
# rfRegressor.fit(X_train, y_train)
#
# # Saving model to disk by serializing the data objects writing it
# pickle.dump(rfRegressor, open('rfModel.pkl', 'wb'))
#
# # Loading model to compare the results (by de-serializing and reading it)
# rfModel = pickle.load(open('rfModel.pkl', 'rb'))
# print(rfModel.predict([[14, 50, 1000, 100]]))