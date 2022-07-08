import pandas as pd


def swap(x):
    x = list(x)
    x[4:6], x[6:8] = x[6:8], x[4:6]
    return ''.join(x)


data = "soccer"
df = pd.read_csv(data + ".csv")
change = df.sample(200).index
df.loc[change, 'stadium'] = df.loc[change, 'stadium'].apply(lambda x: x.replace(x[2:4], ""))
change = df.sample(400).index
df.loc[change, 'stadium'] = df.loc[change, 'stadium'].apply(swap)
change = df.sample(321).index
df.loc[change, 'stadium'] = df.loc[change, 'stadium'].apply(lambda x: x.replace(x[0], "*"))
change = df.sample(950).index
df.loc[change, 'name'] = df.loc[change, 'name'].apply(lambda x: x.replace(x[3], "#"))
change = df.sample(750).index
df.loc[change, 'birthplace'] = df.loc[change, 'birthplace'].apply(lambda x: x.replace(x[0], "55"))
df.to_csv("aaaaa.csv", index=False)
