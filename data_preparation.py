import pandas as pd

def load_data():
    true = pd.read_csv("data/true.csv")
    fake = pd.read_csv("data/fake.csv")

    true['label'] = 1  # Real
    fake['label'] = 0  # Fake

    df = pd.concat([true, fake])
    df.dropna(inplace=True)
    print(f" Loaded {len(df)} articles.")
    print(df.head())
    return df
