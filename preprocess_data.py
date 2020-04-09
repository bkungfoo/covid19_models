import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse
import matplotlib.pyplot as plt
import math
import os
import pickle

START_DATE = datetime(2019, 1, 1)  # Set some long time in the past before infection.


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

    
def get_county(string):
    """Parse out a US county from a combined string containing county, state, and country.

    Args:
      string: The full string in format "county, state, country".
    Returns:
      The county name if it exists, otherwise empty string.
    """
    str_array = string.split(',')
    if len(str_array) < 3:
        return ''
    else:
        return str_array[0]


def write_us_county_data(us_confirmed_df, us_deaths_df, county_census_df, filename):
    column_names = ['FIPS', 'County', 'Province_State', 'Country_Region', 'Date', 'Cases', 'Deaths', 'Population']

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            us_combined_df = pickle.load(f)
    else:
        us_combined_df = pd.DataFrame(columns=column_names)

    if len(us_combined_df):
        start_date = us_combined_df['Date'].max() + timedelta(0, 0, 1)
    else:
        start_date = datetime(2019, 1, 1)

    # Use date conversions to extract correct dates from column names in COVID dataset
    date_cols = [x for x in list(us_confirmed_df) if is_date(x)]
    dates = [datetime.strptime(x, '%m/%d/%y') for x in date_cols]
    dates = [x for x in dates if x >= start_date]
    date_cols = [x for x in date_cols if datetime.strptime(x, '%m/%d/%y') >= start_date]

    # Append rows that have confirmed cases, deaths, and populations included.
    for index, row in us_confirmed_df.iterrows():
        fips = row['FIPS']
        county = get_county(row['Combined_Key'])
        if math.isnan(fips):
            print('skipping county', county, fips, population)
            continue
        population = county_census_df[
            (county_census_df.STATE == int(fips / 1000))
            & (county_census_df.COUNTY == int(fips % 1000))]['POPESTIMATE2019']
        if len(population) != 1:
            print('skipping county', county, fips, population)
            continue
        population = population.to_numpy()[0]
        for (date_col, date) in zip(date_cols, dates):
            confirmed = row[date_col]
            if confirmed == 0:
                continue
            if date_col in us_deaths_df:
                deaths = us_deaths_df[us_deaths_df.FIPS == row['FIPS']][date_col].to_numpy()[0]
            else:
                deaths = 0

            values = [fips, county, row['Province_State'], row['Country_Region'], date, confirmed, deaths, population]
            df_length = len(us_combined_df)
            us_combined_df.loc[df_length] = values
        if index % 100 == 0:
            print('processed {} out of {}'.format(index, len(us_confirmed_df)))
    us_combined_df = us_combined_df.drop_duplicates(['Date', 'FIPS'], keep='last')  # Drop duplicates just in case
    with open(filename, 'wb') as f:
        pickle.dump(us_combined_df, f)


def write_world_country_data(world_confirmed_df, world_deaths_df, world_population_df, filename):
    column_names = ['Country_Region', 'Date', 'Cases', 'Deaths', 'Population']

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            world_combined_df = pickle.load(f)
    else:
        world_combined_df = pd.DataFrame(columns=column_names)

    if len(world_combined_df):
        start_date = world_combined_df['Date'].max() + timedelta(0, 0, 1)
    else:
        start_date = datetime(2019, 1, 1)

    # Use date conversions to extract correct dates from column names in COVID dataset
    date_cols = [x for x in list(world_confirmed_df) if is_date(x)]
    dates = [datetime.strptime(x, '%m/%d/%y') for x in date_cols]
    dates = [x for x in dates if x >= start_date]
    date_cols = [x for x in date_cols if datetime.strptime(x, '%m/%d/%y') >= start_date]

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


def preprocess_data():
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
    preprocess_data()
