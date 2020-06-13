# %%
#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
from pre_process import create_df, df_to_dataset

MODEL_SAVE_PATH = './saved_models/'

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
num_features = ['Pclass', 'SibSp', 'Parch', 'Age']
# num_features = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']
cat_features_freqFill = ['Sex',  'Embarked', 'Ticket', 'Last_Name']
cat_features_xFill = ['Cabin']

all_features = num_features + cat_features_freqFill + cat_features_xFill

train_df = create_df(train_data, num_features, cat_features_freqFill, cat_features_xFill, label)
submission_df = create_df(submission_data, num_features, cat_features_freqFill, cat_features_xFill)

# Descriptives
train_df.corr()

# Test/Train
train, test = train_test_split(train_df, test_size=0.2, random_state=42)

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

train_df.groupby('Ticket').nunique()  # unique tickets = 681
ticket_hashed = feature_column.categorical_column_with_hash_bucket(
      'Ticket', hash_bucket_size=681)
feature_columns.append(feature_column.indicator_column(ticket_hashed))

train_df.groupby('Last_Name').nunique() # unique last names = 667
lname_hashed = feature_column.categorical_column_with_hash_bucket(
      'Last_Name', hash_bucket_size=667)
feature_columns.append(feature_column.indicator_column(lname_hashed))

train_df.groupby('Cabin').nunique() # unique Cabins = 147
cabin_hashed = feature_column.categorical_column_with_hash_bucket(
      'Cabin', hash_bucket_size=147)
feature_columns.append(feature_column.indicator_column(cabin_hashed))

# create feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# Training settings
BATCH_SIZE = 32
EPOCS = 70
LEARNING_RATE = 0.0001
L2 = 1e-4
train_ds = df_to_dataset(train, True, BATCH_SIZE, label)
test_ds = df_to_dataset(test, False, BATCH_SIZE, label)
submission_ds = df_to_dataset(submission_df, False, BATCH_SIZE, has_label=False)


# %%
# Train Model

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
MODEL_NAME = 'class_age_sib_par_fare_sex_emb_ticket_lname_cabin'
model.save(MODEL_SAVE_PATH+MODEL_NAME)

# %%
# Load Model
MODEL_NAME = 'class_age_sib_par_fare_sex_emb_ticket_lname_cabin'
model = tf.keras.models.load_model(MODEL_SAVE_PATH+MODEL_NAME)


# %%
# Get Predictions
SAVE_NAME = './submissions/6_13_submission.csv'
predictions = model.predict_classes(submission_ds)

output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': np.ravel(predictions)})
output.to_csv(SAVE_NAME, index=False)
