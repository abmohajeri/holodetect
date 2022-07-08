import pandas as pd

data = "food"
df = pd.read_csv(data + ".csv")
change = df.sample(189).index
df.loc[change, 'DBA_Name'] = "**" + df.loc[change, 'DBA_Name']
change = df.sample(489).index
df.loc[change, 'State'] = df.loc[change, 'State'] + "xa"
change = df.sample(890).index
df.loc[change, 'Results'] = df.loc[change, 'Results'] + "ee"
df.to_csv("aaaaa.csv", index=False)
