import pandas as pd
import numpy as np

df = pd.read_csv('./Training_Board9.csv', names=range(999))
df.dropna(axis=1, inplace=True)
x = df.values
np.save('./x9_tr.npy', x)


