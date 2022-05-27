import numpy as np
import pandas as pd
from typing import Union

"""
Retrieves labels for data from dataframe.

:param df: dataframe for the data
:returns: true labels for data
:rtype: 1-D numpy array
"""
def retrieve_labels(df: pd.DataFrame) -> Union[np.ndarray, np.ndarray]:
    labels = df.loc[:, 'diagnosis'].to_numpy()

    # Turning characters into digits
    for i in range(len(labels)):
        if labels[i] == 'M':
            labels[i] = 1
        else:
            labels[i] = 0
    return labels[:284], labels[284:]


"""
Removes unnecessary columns from the dataframe.

:param df: pandas dataframe of data.
"""
def prune_columns(df: pd.DataFrame) -> np.ndarray:

    df.drop('Unnamed: 32', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df.loc[:, 'diagnosis'] = 1


""" 
Splits features up into train and test set.

:param features: features of entire dataset
"""
def split_features(features: np.ndarray) -> Union[np.ndarray, np.ndarray]:
    train = features[:284]
    test = features[284:]
    return train, test