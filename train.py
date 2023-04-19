import pandas as pd


data = pd.read_csv('./data/covtype.data', sep=',', header=None)


print(data.head())