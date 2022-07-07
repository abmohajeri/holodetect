import pandas as pd

data = "food"
df = pd.read_csv(data + ".csv")
change = df.sample(890).index
df.loc[change, 'DBA Name'] = df.loc[change, 'DBA Name'] + "x"
change = df.sample(890).index
df.loc[change, 'State'] = df.loc[change, 'State'] + "xa"
change = df.sample(890).index
df.loc[change, 'Facility Type'] = df.loc[change, 'Facility Type'] + "*"
change = df.sample(890).index
df.loc[change, 'Address'] = "x" + df.loc[change, 'Address']
change = df.sample(890).index
df.loc[change, 'City'] = "x" + df.loc[change, 'City']
change = df.sample(890).index
df.loc[change, 'Results'] = df.loc[change, 'Results'] + "ee"
df.to_csv("aaaaa.csv", index=False)
