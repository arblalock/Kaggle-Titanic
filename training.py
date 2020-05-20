import pandas as pd
import numpy as np
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Exploring
train.describe()
train.groupby('Ticket').nunique()
train.isnull().sum()

train.head()

# Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Survived']
train_feat = train[features]
train_feat = train_feat.dropna()


# Descriptives
train_feat.corr()

# Test/Train
train, test = train_test_split(train_feat, test_size=0.2, random_state=42)


def df_to_dataset(dataframe, shuffle, batch_size):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Survived')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


feature_columns = []

# numeric cols
for header in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    feature_columns.append(feature_column.numeric_column(header))


# indicator cols
sex = feature_column.categorical_column_with_vocabulary_list(
    'Sex', ['male', 'female'])
sex_one_hot = feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

# create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# datasets
batch_size = 32
train_ds = df_to_dataset(train, True, batch_size)
test_ds = df_to_dataset(test, False, batch_size)

# model
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train           
history = model.fit(train_ds,
          validation_data=test_ds,
          epochs=30)

# plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# metrics
loss, accuracy = model.evaluate(test_ds)
y_pred = model.predict_classes(test_ds)
con_mat = tf.math.confusion_matrix(labels=test.Survived, predictions=y_pred).numpy()
con_mat = confusion_matrix(test.Survived, y_pred)

# Note 0 = 0 ( meaning died, so in both test and predicted more people died then survived)
test.Survived.sum()