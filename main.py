import pandas as pd
import numpy as np
import preprocessing as p
import model as m

df = pd.read_csv('data.csv')
train_labels, test_labels = p.retrieve_labels(df)
p.prune_columns(df)
features = df.to_numpy()
train, test = p.split_features(features)

model = m.LogisticRegression(train, test, train_labels, test_labels)
model.train()
model.test()