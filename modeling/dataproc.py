"""A class that holds all imported raw data files and contains methods to pull out requested pandas dataframes."""
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle
import os


class DataStore:

    def __init__(self, us_df_path='./data/us_combined_df.pkl',
                 world_df_path='./data/world_combined_df.pkl',
                 us_county_census='./data/co-est2019-alldata.csv'):

        print(os.listdir('./'))
        with open('./data/us_combined_df.pkl', 'rb') as f:
            self.us_combined_df = pickle.load(f)

        # Load covid and world population data
        with open('./data/world_combined_df.pkl', 'rb') as f:
            self.world_combined_df = pickle.load(f)

            self.county_census_df = pd.read_csv('./data/co-est2019-alldata.csv', encoding='latin-1')

    def get_time_series_for_area(self, country, states_and_counties):
        """Slice the dataframe by the are of interest and aggregate confirmed cases and deaths by date.

        NOTE: this function only works if you have already prefetched all of the data in the above cell!

        Args:
            country: A string representing the country
            states_and_counties: If the US, a list of FIPS [(state1, list_of_counties1), (state2, list_of_counties2)].

        Returns:
          A Dataframe holding a time series of confirmed cases and deaths for the area of interest.
        """
        if country == 'US' and states_and_counties is not None:
            agg_df = pd.DataFrame(columns=['Cases', 'Deaths'])
            population = 0
            for state, counties in states_and_counties:
                if not counties:
                    population += self.county_census_df[(self.county_census_df.STATE == state)
                                                        & (self.county_census_df.COUNTY == 0)]['POPESTIMATE2019'].sum()
                    area_df = self.us_combined_df[
                        (self.us_combined_df.FIPS > state * 1000)
                        & (self.us_combined_df.FIPS < (state + 1) * 1000)].groupby('Date').agg(
                        {'Cases': 'sum',
                         'Deaths': 'sum',
                         })
                else:
                    combined_fips = [state * 1000 + y for y in counties]
                    population += self.county_census_df[
                        (self.county_census_df.STATE == state)
                        & (self.county_census_df.COUNTY.isin(counties))]['POPESTIMATE2019'].sum()
                    area_df = self.us_combined_df[
                        (self.us_combined_df.FIPS.isin(combined_fips))].groupby('Date').agg({
                        'Cases': 'sum',
                        'Deaths': 'sum',
                    })
                agg_df = agg_df.join(area_df, how='outer', rsuffix='_other')
                agg_df = agg_df.fillna(0)
                agg_df['Cases'] = agg_df['Cases'] + agg_df['Cases_other']
                agg_df['Deaths'] = agg_df['Deaths'] + agg_df['Deaths_other']
                del agg_df['Cases_other']
                del agg_df['Deaths_other']
        else:
            agg_df = self.world_combined_df[
                (self.world_combined_df.Country_Region == country)]
            population = agg_df['Population'].iloc[0]
        agg_df = agg_df.reset_index()
        agg_df = agg_df.drop_duplicates(['Date'], keep='last')
        agg_df = agg_df.sort_values('Date').reset_index()
        return agg_df, population


def detrend_day_of_week(bounded_area_df, area_df, metric, weeks=4):
    # Columns
    diff_col = metric + ' diff'
    diff_sq_col = metric + ' squared diff'
    diff_adj_col = metric + ' diff adjusted'
    adj_cumsum_col = metric + ' adjusted cumsum'

    # Original DF with all dates.
    area_df['dow'] = area_df['Date'].dt.dayofweek
    area_df[diff_col] = area_df[metric] - area_df[metric].shift(1)
    area_df[diff_sq_col] = area_df[diff_col] * area_df[diff_col]
    dow_sum = None
    weeks = 4

    weekly_sum = 0
    weekly_sum_squared = 0
    for days_before in range(weeks * 7 - 1, weeks * 7 + 6):  # 7 days shift
        for week in range(weeks):
            start_date = area_df['Date'].max() + timedelta(-days_before + week * 7)
            norm_dow = area_df[
                (area_df['Date'] >= start_date) &
                (area_df['Date'] <= start_date + timedelta(6))
                ][['Date', metric, diff_col, diff_sq_col, 'dow']].groupby('dow').sum()

            dow_mean = norm_dow[diff_col].mean()
            norm_dow[diff_col] /= dow_mean
            norm_dow[diff_sq_col] /= (dow_mean ** 2)
            weekly_sum += norm_dow[diff_col].sum()
            weekly_sum_squared += norm_dow[diff_sq_col].sum()
            if dow_sum is None:  # 1 week of data
                dow_sum = norm_dow
            else:
                dow_sum += norm_dow
    # Total number of samples is weeks * 7, so we take the mean by dividing by weeks * 7
    death_mean = dow_sum[diff_col] / weeks / 7
    death_err = np.sqrt(dow_sum[diff_sq_col] / weeks / 7 - death_mean * death_mean) / np.sqrt(weeks)
    weekly_mean = weekly_sum / weeks / 7 / 7
    weekly_err = np.sqrt(weekly_sum_squared / weeks / 7 / 7 - weekly_mean * weekly_mean) / np.sqrt(weeks)
    # Apply Gaussian weighted averaging (Kalman Filter)
    kf_mean = (death_mean * (1 / death_err ** 2) + weekly_mean * (1 / weekly_err ** 2)
               ) / ((1 / death_err ** 2) + (1 / weekly_err ** 2))
    if np.isnan(kf_mean).any() or (kf_mean == 0).any():
        print('skipping dow detrending')
        bounded_area_df[adj_cumsum_col] = bounded_area_df[metric]  # Don't do any day of week smoothing
        return bounded_area_df
    # DF to modify.
    bounded_area_df['dow'] = bounded_area_df['Date'].dt.dayofweek
    bounded_area_df[diff_col] = bounded_area_df[metric] - bounded_area_df[metric].shift(1)
    bounded_area_df[diff_adj_col] = bounded_area_df[diff_col] / bounded_area_df['dow'].apply(lambda x: kf_mean[x])
    bounded_area_df[diff_adj_col].iloc[0] = 0
    bounded_area_df[adj_cumsum_col] = bounded_area_df[diff_adj_col].cumsum()
    bounded_area_df[adj_cumsum_col] = bounded_area_df[adj_cumsum_col] + bounded_area_df[metric].iloc[0]

    return bounded_area_df


def convert_data_to_numpy(area_df, metric, smooth=True, window_start=-3, window_end=3):
    """Convert the metric to model into a numpy array, with the option to apply smoothing.

    If smooth=True, this will use a window to remove single day outliers as follows:
      If the slope (metric(t) / metric(t-1) for a given day exceeds the min or max of all other slopes in the window,
      the slope will be constrainted to the min or max of all other slopes.

    Note that this filter is different from the average smoothing used in http://www.healthdata.org/covid/updates,
    as it explicitly removes outliers rather than averaging nearby points, which can skew toward outliers. It is also
    less restrictive than the median, and so can react more quickly to changes in slope.

    Args:
        area_df: A dataframe of confirmed cases and deaths returned by the
        metric: Either "Cases" or "Deaths"
        smooth: Apply smoothing filter
        window_start: Starting offset from current day for the window to use to remove single day outlier.
        window_end: Ending offset from current day for the window to use to remove single day outlier.
    """
    data = area_df[metric].to_numpy()
    if smooth:
        smoothed_deaths = []
        for index in range(data.shape[0]):
            if len(smoothed_deaths) <= window_end - window_start or smoothed_deaths[
                                index - window_end + window_start] == 0:
                smoothed_deaths.append(data[index])
            else:
                # Get exponential slopes and remove outlier
                slopes = (
                    data[index + window_start + 1:min(data.shape[0], index + window_end)]
                    / data[index + window_start:min(data.shape[0], index + window_end) - 1]
                )
                curr_slope = slopes[-window_start - 1]
                slopes[-window_start - 1] = np.median(slopes)  # Effectively remove the current slope from the list.
                if curr_slope > np.max(slopes):
                    curr_slope = np.max(slopes)
                elif curr_slope < np.min(slopes):
                    curr_slope = np.min(slopes)
                smoothed_deaths.append(smoothed_deaths[-1] * curr_slope)

        smoothed_deaths = np.array(smoothed_deaths)
        # TODO: Not sure if this final normalization is helpful.
        return smoothed_deaths * area_df[metric].sum() / (np.sum(smoothed_deaths) + 1e-4)
    else:
        return data