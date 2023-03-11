import pandas as pd

temp = pd.read_csv('../res/BACE/summary0223.csv').drop_duplicates()
temp.to_csv('../res/BACE/summary0223.csv',index=False)

temp = pd.read_csv('../res/BBBP/summary0223.csv').drop_duplicates()
temp.to_csv('../res/BBBP/summary0223.csv',index=False)