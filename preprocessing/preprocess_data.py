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


def _filter_world_cols(df, metric):
    del df['Province/State']
    del df['Lat']
    del df['Long']
    df = df.rename(columns={'Country/Region': 'country'})
    df = df.groupby(['country']).agg('sum').reset_index()
    df = df.melt(id_vars=['country'], var_name='Date', value_name=metric)
    return df


def write_us_county_data(us_confirmed_df, us_deaths_df, us_county_census_df, filename):
    us_confirmed_df = _filter_us_cols(us_confirmed_df, 'Cases')
    us_deaths_df = _filter_us_cols(us_deaths_df, 'Deaths')
    us_deaths_df = us_deaths_df[['FIPS', 'Date', 'Deaths']]
    us_deaths_df = us_deaths_df.set_index(['FIPS', 'Date'])
    us_county_census_df['FIPS'] = us_county_census_df.STATE * 1000 + us_county_census_df.COUNTY
    us_county_census_df = us_county_census_df.set_index('FIPS')
    us_confirmed_df = us_confirmed_df.merge(us_county_census_df[['CTYNAME', 'STNAME', 'POPESTIMATE2019']], on='FIPS',
                                           how='inner')
    us_confirmed_df = us_confirmed_df.merge(us_deaths_df, on=['FIPS', 'Date'], how='inner')
    us_confirmed_df['Date'] = us_confirmed_df['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%y'))
    with open(filename, 'wb') as f:
        pickle.dump(us_confirmed_df, f)


def write_world_country_data(world_confirmed_df, world_deaths_df, world_population_df, filename):
    column_names = ['Country_Region', 'Date', 'Cases', 'Deaths', 'Population']
    world_confirmed_df = _filter_world_cols(world_confirmed_df, 'Cases')
    world_deaths_df = _filter_world_cols(world_deaths_df, 'Deaths')
    world_combined_df = world_confirmed_df.merge(world_population_df, on='country', how='inner')
    world_combined_df = world_combined_df.merge(world_deaths_df, on=['country', 'Date'], how='inner')
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
    write_world_country_data(world_confirmed_df, world_deaths_df, world_population_df, './data/world_combined_df.pkl')


if __name__ == "__main__":
    app.run(preprocess_data)
