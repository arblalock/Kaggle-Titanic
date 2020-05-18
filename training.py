import pandas as pd
import numpy as np
from numpy import nan
from sklearn.model_selection import train_test_split

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

#Exploring
train.describe()
train.groupby('Ticket').nunique()
train.isnull().sum()


#Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Survived']
train_feat = train[features]
train_feat = train_feat.dropna()


#Descriptives
train_feat.corr()

#Test/Train
y = train_feat.pop('Survived')
X = train_feat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

