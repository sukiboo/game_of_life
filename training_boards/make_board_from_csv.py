import pandas as pd
import numpy as np


def x4_mirror(df):
    x4 = np.array(df)
    x4 = np.concatenate(
            [
                x4,
                np.flip(x4, axis=0)
            ],
            axis=0
        )
    x4 = np.concatenate(
            [
                x4,
                np.flip(x4, axis=1)
            ],
            axis=1
        )
    return pd.DataFrame(x4)


if __name__ == '__main__':

    df = pd.read_csv('./Training_Board.csv', names=range(999))
    df.dropna(axis=1, inplace=True)
    df = x4_mirror(df)
    x = df.values
    np.save('./fixed.npy', x)


