import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse
import matplotlib.pyplot as plt
import math
import os
import pickle

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('start_date', '', 'Start date for importing data. Format is YYYY-MM-dd. If not specified, '
                    'will default to creating the entire table if the table does not already exist, and doing '
                    'incremental updates if the table already exists.')

flags.DEFINE_string('end_date', '', 'End date for importing data. Format is YYYY-MM-dd. If not specified, '
                    'will default to the very last date available in imported data.')

DEFAULT_START_DATE = datetime(2019, 1, 1)  # Set some long time in the past before infection.
DEFAULT_END_DATE = datetime(2100, 1, 1)  # Set some long time in the future after infection.


def is_date(string, fuzzy=False):
    """Return whether the string can be interpreted as a date.

    Args:
      string: String to check for date.
    Args:
      fuzzy: Ignore unknown tokens in string if True.
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def _filter_us_cols(df, metric):
    df = df[(df['FIPS'] >= 0) & (~df['FIPS'].isna())]
    del df['Admin2']
    del df['iso2']
    del df['UID']
    del df['iso3']
    del df['code3']
    del df['Province_State']
    del df['Country_Region']
    try:
        del df['Population']
    except:
        print('Population does not exist')
    try:
        del df['Combined_Key']
    except:
        print('Combined key does not exist')
    df = df.melt(id_vars=['FIPS', 'Lat', 'Long_'],
                 var_name='Date', value_name=metric)
    return df


def write_us_county_data(us_confirmed_df, us_deaths_df, us_county_census_df, filename):
    us_confirmed_df = _filter_us_cols(us_confirmed_df, 'Cases')
    us_deaths_df = _filter_us_cols(us_deaths_df, 'Deaths')
    us_deaths_df = us_deaths_df[['FIPS', 'Date', 'Deaths']]
    us_deaths_df = us_deaths_df.set_index(['FIPS', 'Date'])
    us_county_census_df['FIPS'] = us_county_census_df.STATE * 1000 + us_county_census_df.COUNTY
    us_county_census_df = us_county_census_df.set_index('FIPS')
    us_confirmed_df = us_confirmed_df.join(us_county_census_df[['CTYNAME', 'STNAME', 'POPESTIMATE2019']], on='FIPS',
                                           how='inner')
    us_confirmed_df = us_confirmed_df.join(us_deaths_df, on=['FIPS', 'Date'], how='inner')
    us_confirmed_df['Date'] = us_confirmed_df['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%y'))
    with open(filename, 'wb') as f:
        pickle.dump(us_confirmed_df, f)


def write_world_country_data(world_confirmed_df, world_deaths_df, world_population_df, filename,
                             start_date='', end_date=''):
    column_names = ['Country_Region', 'Date', 'Cases', 'Deaths', 'Population']

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            world_combined_df = pickle.load(f)
    else:
        world_combined_df = pd.DataFrame(columns=column_names)

    if not start_date:
        if len(world_combined_df):
            start_date = world_combined_df['Date'].max() + timedelta(0, 0, 1)
        else:
            start_date = datetime(2019, 1, 1)
    else:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    if not end_date:
        end_date = DEFAULT_END_DATE
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Use date conversions to extract correct dates from column names in COVID dataset
    date_cols = [x for x in list(world_confirmed_df) if is_date(x)]
    dates = [datetime.strptime(x, '%m/%d/%y') for x in date_cols]
    dates = [x for x in dates if x >= start_date and x <= end_date]
    date_cols = [x for x in date_cols
                 if datetime.strptime(x, '%m/%d/%y') >= start_date
                 and datetime.strptime(x, '%m/%d/%y') <= end_date
                ]

    # Aggregate regions for countries
    world_confirmed_agg_df = world_confirmed_df.groupby(['Country/Region']).agg({
        d: 'sum' for d in date_cols
    }).reset_index()
    world_deaths_agg_df = world_deaths_df.groupby(['Country/Region']).agg({
        d: 'sum' for d in date_cols
    }).reset_index()

    # Start populating combined dataframe to pickle
    for index, row in world_confirmed_agg_df.iterrows():
        country = row['Country/Region']
        population = world_population_df[world_population_df['country'] == row['Country/Region']]['Population']
        if len(population) != 1:
            print('skipping country', country, population)
            continue
        population = population.to_numpy()[0]
        for (date_col, date) in zip(date_cols, dates):
            confirmed = row[date_col]
            if confirmed == 0:
                continue
            if date_col in world_deaths_agg_df:
                deaths = world_deaths_agg_df[
                    (world_deaths_agg_df['Country/Region'] == row['Country/Region'])
                ][date_col].to_numpy()[0]
            else:
                deaths = 0

            values = [country, date, confirmed, deaths, population]
            df_length = len(world_combined_df)
            world_combined_df.loc[df_length] = values
        if index % 100 == 0:
            print('processed {} out of {}'.format(index, len(world_confirmed_agg_df)))
    with open(filename, 'wb') as f:
        pickle.dump(world_combined_df, f)


def preprocess_data(argv):
    # Get covid confirmed cases and deaths for each US county and date.
    us_confirmed_df = pd.read_csv(
        '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    us_deaths_df = pd.read_csv(
        '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
    # Gather covid confirmed cases and deaths for each country and date.
    world_confirmed_df = pd.read_csv(
        '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    world_deaths_df = pd.read_csv(
        '../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    # US County level census data
    # Modified from
    # https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv
    us_county_census_df = pd.read_csv('./data/co-est2019-alldata.csv', encoding='latin-1')

    # World population data (for MOST countries).
    world_population_df = pd.read_csv('./data/country_profile_variables.csv')
    world_population_df = world_population_df[['country', 'Population in thousands (2017)']]
    world_population_df = world_population_df.rename(columns={'Population in thousands (2017)': 'Population'})
    # Add some missing countries
    world_population_df.loc[len(world_population_df)] = ['Taiwan*', 23780]
    world_population_df['Population'] *= 1000

    # Write to disk
    write_us_county_data(us_confirmed_df, us_deaths_df, us_county_census_df, './data/us_combined_df.pkl')
    write_world_country_data(world_confirmed_df, world_deaths_df, world_population_df, './data/world_combined_df.pkl',
                             FLAGS.start_date, FLAGS.end_date)


if __name__ == "__main__":
    app.run(preprocess_data)
