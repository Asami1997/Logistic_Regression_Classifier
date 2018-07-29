# Data Pre-Processing

# Importing Libraries 
import numpy as np
import pandas as pd
import matplotlib as mat

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# extract independent variables. will return a matrix
x = dataset.iloc[:, 2:4].values
# extract dependent varaibles. will return a vector
y = dataset.iloc[: , -1].values

# Spliting dataset into training and testing set
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train, y_test = train_test_split(x , y, test_size=0.25, random_state=0)

# Feature scaling 
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test  = x_scaler.fit_transform(x_test)


# Fitting logistic regression into training set
from sklearn.linear_model import LogisticRegression
#This is where it learns the coorelation between x_train and y_train
# THe classifier will learn after executing this line
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predict the test set results
# y_pred is vector that will contain the predictions of each of the test set observations
y_pred = classifier.predict(x_test)

# Use confusion matrix for evaluation 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the results
from matplotlib.colors import ListedColormap