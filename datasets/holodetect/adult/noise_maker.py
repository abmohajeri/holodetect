import pandas as pd


def swap(x):
    x = list(x)
    x[4:6], x[6:8] = x[6:8], x[4:6]
    return ''.join(x)


data = "adult"
df = pd.read_csv(data + ".csv")
change = df.sample(200).index
df.loc[change, 'income'] = df.loc[change, 'income'].apply(lambda x: x.replace(x[4:6], ""))
change = df.sample(100).index
df.loc[change, 'income'] = df.loc[change, 'income'].apply(swap)
change = df.sample(321).index
df.loc[change, 'income'] = df.loc[change, 'income'] + "**"
df.to_csv("aaaaa.csv", index=False)
