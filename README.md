# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Open the csv file and form a dataframe using pandas library
2. Split train and test data from the dataset
3. Initiate LinearRegression model and fit the Training dataset
4. Plot the Predicted Line using matplotlib

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv("./content/student_marks.csv")
df.head()
plt.scatter(df["X"],df["Y"])
plt.xlabel("X")
plt.ylabel("y")
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
plt.scatter(df["X"],df["Y"])
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_train,lr.predict(X_train),color="red");
*/
```

## Output:
![image](https://github.com/Bhargava-123/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/85554376/2ffa5038-8339-47ac-a1c4-c06eba58d726)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
