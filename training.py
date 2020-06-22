# %%
#Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
from pre_process import create_df, df_to_dataset
%load_ext autoreload
%autoreload 2

os.chdir('/workspaces/MachineLearning/Kaggle/Titanic/')

MODEL_SAVE_PATH = './saved_models/'
RAND = 10
STANDARDIZE = True
TEST_SIZE=0.2

# %%
#Import data
train_data = pd.read_csv('./data/train.csv')
submission_data = pd.read_csv('./data/test.csv')

# Exploring
train_data.describe()
train_data.head()
print(train_data.isnull().sum())

# Feature selection/creation
label = 'Survived'
num_features = ['SibSp', 'Parch', 'Age', 'Fare']
# num_features = ['Age', 'Fare']
cat_features_freqFill = ['Sex', 'Embarked', 'Pclass', 'Ticket', 'Last_Name']
# cat_features_freqFill = ['Sex',  'Embarked', 'Ticket', 'Last_Name']
cat_features_xFill = []
# cat_features_xFill = ['Cabin']

all_features = num_features + cat_features_freqFill + cat_features_xFill

train_df = create_df(train_data, num_features, cat_features_freqFill, cat_features_xFill, STANDARDIZE, label)
submission_df = create_df(submission_data, num_features, cat_features_freqFill, cat_features_xFill, STANDARDIZE)

# Descriptives
train_df.corr()

# Test/Train
train, test = train_test_split(train_df, test_size=TEST_SIZE, random_state=RAND)


feature_columns = []

# numeric cols
for f in num_features:
    feature_columns.append(feature_column.numeric_column(f))


# indicator cols
if 'Sex' in cat_features_freqFill:
    sex = feature_column.categorical_column_with_vocabulary_list(
        'Sex', ['male', 'female'])
    feature_columns.append(feature_column.indicator_column(sex))

if 'Pclass' in cat_features_freqFill:
    Pclass = feature_column.categorical_column_with_vocabulary_list(
        'Pclass', [1, 2, 3])
    feature_columns.append(feature_column.indicator_column(Pclass))

if 'Embarked' in cat_features_freqFill:
    embarked = feature_column.categorical_column_with_vocabulary_list(
        'Embarked', ['C', 'Q', 'S'])
    feature_columns.append(feature_column.indicator_column(embarked))

if 'Ticket' in cat_features_freqFill:
    train_df.groupby('Ticket').nunique()  # unique tickets = 681
    ticket_hashed = feature_column.categorical_column_with_hash_bucket(
        'Ticket', hash_bucket_size=681)
    feature_columns.append(feature_column.indicator_column(ticket_hashed))

if 'Last_Name' in cat_features_freqFill:
    train_df.groupby('Last_Name').nunique() # unique last names = 667
    lname_hashed = feature_column.categorical_column_with_hash_bucket(
        'Last_Name', hash_bucket_size=667)
    feature_columns.append(feature_column.indicator_column(lname_hashed))

if 'Cabin' in cat_features_xFill:
    train_df.groupby('Cabin').nunique() # unique Cabins = 147
    cabin_hashed = feature_column.categorical_column_with_hash_bucket(
        'Cabin', hash_bucket_size=147)
    feature_columns.append(feature_column.indicator_column(cabin_hashed))

sib_col = feature_column.numeric_column('SibSp')
parch_col = feature_column.numeric_column('Parch')
age_col = feature_column.numeric_column('Age')
fare_col = feature_column.numeric_column('Fare')
age_buckets = feature_column.bucketized_column(fare_col, boundaries=[0, 18, 20, 30, 40, 50, 60, 70])
fare_buckets = feature_column.bucketized_column(age_col, boundaries=[0, 5, 10, 20, 30, 50, 100, 200, 300, 400])
sib_buckets = feature_column.bucketized_column(sib_col, boundaries=list(np.arange(0, 6, step=1)))
parch_buckets = feature_column.bucketized_column(parch_col, boundaries=list(np.arange(0, 6, step=1)))

age_fare_cross = feature_column.crossed_column([age_buckets, fare_buckets], hash_bucket_size=1000)
feature_columns.append(feature_column.indicator_column(age_fare_cross))


# sex_fare_cross = feature_column.crossed_column([sex, fare_buckets], hash_bucket_size=1000)
# feature_columns.append(feature_column.indicator_column(sex_fare_cross))

# fare_pclass_cross = feature_column.crossed_column([fare_buckets, Pclass], hash_bucket_size=1000)
# feature_columns.append(feature_column.indicator_column(fare_pclass_cross))

# embarked_fare_cross = feature_column.crossed_column([fare_buckets, embarked], hash_bucket_size=100)
# feature_columns.append(feature_column.indicator_column(embarked_fare_cross))

# age_sib_cross = feature_column.crossed_column([age_buckets, sib_buckets], hash_bucket_size=1000)
# feature_columns.append(feature_column.indicator_column(age_sib_cross))

# age_parch_cross = feature_column.crossed_column([age_buckets, parch_buckets], hash_bucket_size=1000)
# feature_columns.append(feature_column.indicator_column(age_parch_cross))

sex_pclass_cross = feature_column.crossed_column([sex, parch_buckets], hash_bucket_size=1000)
feature_columns.append(feature_column.indicator_column(sex_pclass_cross))

# create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# %%
# Train Model
# Training settings
BATCH_SIZE = 80
EPOCS = 40
LEARNING_RATE = 0.0001
L2 = 1e-4
train_ds = df_to_dataset(train, True, BATCH_SIZE, label)
test_ds = df_to_dataset(test, False, BATCH_SIZE, label)
submission_ds = df_to_dataset(submission_df, False, BATCH_SIZE, has_label=False)

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# model
model = tf.keras.Sequential([
  feature_layer,
  layers.Dropout(0.2),
  layers.Dense(100, activation='relu', activity_regularizer=regularizers.l2(L2)),
  layers.Dense(100, activation='relu', activity_regularizer=regularizers.l2(L2)),
  layers.Dense(100, activation='relu', activity_regularizer=regularizers.l2(L2)),
  layers.Dense(100, activation='relu', activity_regularizer=regularizers.l2(L2)),
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

# Evaluate
loss, accuracy = model.evaluate(test_ds)
y_pred = model.predict_classes(test_ds)
y_true = test.Survived
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
target_names = ['did not survive', 'survived']
print(classification_report(y_true, y_pred, target_names=target_names))

# %%
# Save Model
MODEL_NAME = 'dropout_withcross'
model.save(MODEL_SAVE_PATH+MODEL_NAME)

# %%
# Load Model
MODEL_NAME = 'class_age_sib_par_fare_sex_emb'
model = tf.keras.models.load_model(MODEL_SAVE_PATH+MODEL_NAME)


# %%
# Get Predictions
SAVE_NAME = './submissions/6_20_submission_dropout_withcross.csv'
predictions = model.predict_classes(submission_ds)

output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': np.ravel(predictions)})
output.to_csv(SAVE_NAME, index=False)
