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
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = './saved_models/'

# %%
#Import data
train_raw = pd.read_csv('./data/train.csv')

last_names = []
for x in train_raw['Name']:
    last_names.append(x.split(',')[0])

train_raw['Last_Name'] = last_names
# Exploring
train_raw.describe()
train_raw.head()
print(train_raw.isnull().sum())



# Feature selection
label = 'Survived'
num_features = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']
cat_features = ['Sex',  'Embarked', 'Ticket', 'Last_Name']


train_ds = train_raw.copy()
train_ds.fillna(np.nan)
num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
cat_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

num_feat_ds = train_ds[num_features]
cat_feat_ds = train_ds[cat_features]
label_ds = train_ds[label]

#Handle NANs
for feat in num_features:
    num_feat_ds[feat] = num_imp.fit_transform(np.array(num_feat_ds[feat]).reshape(-1, 1))

for feat in cat_features:
    cat_feat_ds[feat] = cat_imp.fit_transform(np.array(cat_feat_ds[feat]).reshape(-1, 1))

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
    'Sex', ['male', 'female'])
feature_columns.append(feature_column.indicator_column(sex))

embarked = feature_column.categorical_column_with_vocabulary_list(
    'Embarked', ['C', 'Q', 'S'])
feature_columns.append(feature_column.indicator_column(embarked))

train_raw.groupby('Ticket').nunique()  # unique tickets = 681
ticket_hashed = feature_column.categorical_column_with_hash_bucket(
      'Ticket', hash_bucket_size=681)
feature_columns.append(feature_column.indicator_column(ticket_hashed))

train_raw.groupby('Last_Name').nunique() # unique last names = 667
lname_hashed = feature_column.categorical_column_with_hash_bucket(
      'Last_Name', hash_bucket_size=667)
feature_columns.append(feature_column.indicator_column(lname_hashed))



# create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# %%
# Train Model
BATCH_SIZE = 32
EPOCS = 80
LEARNING_RATE = 0.0001
L2 = 1e-3
train_ds = df_to_dataset(train, True, BATCH_SIZE)
test_ds = df_to_dataset(test, False, BATCH_SIZE)


# example_batch = next(iter(train_ds))[0]
# def demo(feature_column):
#   feature_layer = layers.DenseFeatures(feature_column)
#   return feature_layer(example_batch).numpy()
# peak = demo(feature_column.indicator_column(ticket_hashed))

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# model
model = tf.keras.Sequential([
  feature_layer,
#   layers.Dense(128, activation='relu'),
#   layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu', activity_regularizer=regularizers.l2(L2)),
  layers.Dense(128, activation='relu', activity_regularizer=regularizers.l2(L2)),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=opt,
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

loss, accuracy = model.evaluate(test_ds)
y_pred = model.predict_classes(test_ds)
y_true = test.Survived
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
target_names = ['did not survive', 'survived']
print(classification_report(y_true, y_pred, target_names=target_names))



# %%
# Save Model
MODEL_NAME = 'class_age_sib_par_fare_sex_emb_ticket_lname'
model.save(MODEL_SAVE_PATH+MODEL_NAME)

# %%
# Load Model
MODEL_NAME = 'class_age_sib_par_fare_sex_embc'
model = tensorflow.keras.models.load_model(MODEL_SAVE_PATH+MODEL_NAME)
