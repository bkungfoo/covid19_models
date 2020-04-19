import pandas as pd
import json

county_census_df = pd.read_csv('./data/co-est2019-alldata.csv', encoding='latin-1')
states_fips_df = county_census_df[['STATE', 'STNAME']].drop_duplicates()

array = []
for index, row in states_fips_df.iterrows():
    array.append({'Country': 'US',
                  'Title': row['STNAME'],
                  'StateFIPS': row['STATE']})

with open('config/states.json', 'w') as f:
    json.dump(array, f)