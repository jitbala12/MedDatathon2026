import pandas as pd
from config import DATA_PATH

def load_wdbc():
    df = pd.read_csv(DATA_PATH, header=None)

    # Remove ID column
    df = df.drop(columns=[0])

    # Convert labels
    df[1] = df[1].map({"M": 1, "B": 0})

    # y is diagnosis X is the features
    y = df[1]
    X = df.drop(columns=[1])

    return X, y