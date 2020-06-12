# %%
#Import libraries
import pandas as pd
import numpy as np
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

# %%
#Import data
train_raw = pd.read_csv('./data/train.csv')

# Exploring
train_raw.describe()
train_raw.groupby('Ticket').nunique()
train_raw.isnull().sum()
train_raw.head()

# Feature selection
label = 'Survived'
# num_features = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']
num_features = ['Pclass', 'SibSp', 'Parch', 'Age']
cat_features = ['Sex']


train_ds = train_raw.copy()
train_ds.fillna(np.nan)
print(train_ds.isnull().sum())
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

num_feat_ds = train_ds[num_features]
cat_feat_ds = train_ds[cat_features]
label_ds = train_ds[label]

#Handle NANs
for feat in num_features:
    num_feat_ds[feat] = imp.fit_transform(np.array(num_feat_ds[feat]).reshape(-1, 1))

train_feat = pd.concat([num_feat_ds, cat_feat_ds, label_ds], axis=1, sort=False)

# Descriptives
train_feat.corr()

# Test/Train
train, test = train_test_split(train_feat, test_size=0.2, random_state=42)


def df_to_dataset(dataframe, shuffle, batch_size):
  dataframe = dataframe.copy()
  labels = dataframe.pop(label)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


feature_columns = []

# numeric cols
for header in num_features:
    feature_columns.append(feature_column.numeric_column(header))


# indicator cols
sex = feature_column.categorical_column_with_vocabulary_list(
    cat_features[0], ['male', 'female'])
sex_one_hot = feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

# create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# %%
# Training
BATCH_SIZE = 32
EPOCS = 200
train_ds = df_to_dataset(train, True, BATCH_SIZE)
test_ds = df_to_dataset(test, False, BATCH_SIZE)

# model
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu', activity_regularizer=regularizers.l2(1e-5)),
  layers.Dense(128, activation='relu', activity_regularizer=regularizers.l2(1e-5)),
  layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train           
history = model.fit(train_ds,
          validation_data=test_ds,
          epochs=EPOCS)

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
y_true = test.Survived
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
target_names = ['did not survive', 'survived']
print(classification_report(y_true, y_pred, target_names=target_names))

# %%
