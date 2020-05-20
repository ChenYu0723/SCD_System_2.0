import pandas as pd

print('Start testing server...')
print('Reading test data...')
df = pd.read_csv('../data/raw_data/metroStations.csv')
print(df)
print('End test')
