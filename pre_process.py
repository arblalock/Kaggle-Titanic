import pandas as pd
import numpy as np
from numpy import nan
from sklearn.impute import SimpleImputer
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def create_df(data, num_features, cat_features_freqFill, cat_features_xFill, standardize, label=[]):
    df = data.copy()
    last_names = []
    for x in df['Name']:
        last_names.append(x.split(',')[0])

    df['Last_Name'] = last_names
    num_feat_ds = df[num_features]
    cat_feat_ds = df[cat_features_freqFill + cat_features_xFill]
    label_ds = df[label]

    #Missing data strategies
    df.fillna(np.nan)
    num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    cat_imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cat_imp_x = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='xxxxx')

    for feat in num_features:
        #Impute
        num_feat_ds[feat] = num_imp.fit_transform(np.array(num_feat_ds[feat]).reshape(-1, 1))


    for feat in cat_features_freqFill:
        cat_feat_ds[feat] = cat_imp_freq.fit_transform(np.array(cat_feat_ds[feat]).reshape(-1, 1))

    for feat in cat_features_xFill:
        cat_feat_ds[feat] = cat_imp_x.fit_transform(np.array(cat_feat_ds[feat]).reshape(-1, 1))

    #Standardize
    if standardize == True and len(num_features) > 0:
        num_feat_sc = StandardScaler().fit_transform(num_feat_ds)
        num_feat_ds = pd.DataFrame(num_feat_sc, columns = num_feat_ds.columns)

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