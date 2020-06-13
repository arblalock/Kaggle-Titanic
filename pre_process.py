import pandas as pd
import numpy as np
from numpy import nan
from sklearn.impute import SimpleImputer
import tensorflow as tf

def create_df(data, num_features, cat_features_freqFill, cat_features_xFill, label=[]):
    df = data.copy()
    last_names = []
    for x in df['Name']:
        last_names.append(x.split(',')[0])

    df['Last_Name'] = last_names
    num_feat_ds = df[num_features]
    cat_feat_ds = df[cat_features_freqFill + cat_features_xFill]
    label_ds = df[label]

    #Handle missing data
    df.fillna(np.nan)
    num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    cat_imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cat_imp_x = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='xxxxx')

    for feat in num_features:
        num_feat_ds[feat] = num_imp.fit_transform(np.array(num_feat_ds[feat]).reshape(-1, 1))


    for feat in cat_features_freqFill:
        cat_feat_ds[feat] = cat_imp_freq.fit_transform(np.array(cat_feat_ds[feat]).reshape(-1, 1))

    for feat in cat_features_xFill:
        cat_feat_ds[feat] = cat_imp_x.fit_transform(np.array(cat_feat_ds[feat]).reshape(-1, 1))

    return pd.concat([num_feat_ds, cat_feat_ds, label_ds], axis=1, sort=False)


def df_to_dataset(dataframe, shuffle, batch_size, label=None, has_label=True):
    dataframe = dataframe.copy()
    if has_label:     
        labels = dataframe.pop(label)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds