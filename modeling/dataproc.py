"""A class that holds all imported raw data files and contains methods to pull out requested pandas dataframes."""
import numpy as np
import pandas as pd
import pickle


class DataStore:

    def __init__(self, us_df_path='./data/us_combined_df.pkl',
                 world_df_path='./data/world_combined_df.pkl',
                 us_county_census='./data/co-est2019-alldata.csv'):

        with open('./data/us_combined_df.pkl', 'rb') as f:
            self.us_combined_df = pickle.load(f)

        # Load covid and world population data
        with open('./data/world_combined_df.pkl', 'rb') as f:
            self.world_combined_df = pickle.load(f)

            self.county_census_df = pd.read_csv('./data/co-est2019-alldata.csv', encoding='latin-1')

    def get_time_series_for_area(self, country, state_fips=None, county_fips=[]):
        """Slice the dataframe by the are of interest and aggregate confirmed cases and deaths by date.

        NOTE: this function only works if you have already prefetched all of the data in the above cell!

        Args:
          area_of_interest_spec: A tuple (country, area_name, state_fips, county_fips).

        Returns:
          A Dataframe holding a time series of confirmed cases and deaths for the area of interest.
        """
        if country == 'US' and state_fips is not None:
            if not county_fips:
                population = self.county_census_df[(self.county_census_df.STATE == state_fips)
                                                   & (self.county_census_df.COUNTY == 0)]['POPESTIMATE2019'].sum()
                area_df = self.us_combined_df[
                    (self.us_combined_df.FIPS > state_fips * 1000)
                    & (self.us_combined_df.FIPS < (state_fips + 1) * 1000)].groupby('Date').agg(
                    {'Cases': 'sum',
                     'Deaths': 'sum',
                     })
            else:
                combined_fips = [state_fips * 1000 + y for y in county_fips]
                population = self.county_census_df[
                    (self.county_census_df.STATE == state_fips)
                    & (self.county_census_df.COUNTY.isin(county_fips))]['POPESTIMATE2019'].sum()
                area_df = self.us_combined_df[
                    (self.us_combined_df.FIPS.isin(combined_fips))].groupby('Date').agg({
                    'Cases': 'sum',
                    'Deaths': 'sum',
                })
        else:
            area_df = self.world_combined_df[
                (self.world_combined_df.Country_Region == country)]
            population = area_df['Population'].iloc[0]
        area_df = area_df.reset_index()
        area_df = area_df.drop_duplicates(['Date'], keep='last')
        area_df = area_df.sort_values('Date').reset_index()
        print('Total population', population)
        return area_df, population


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
        return smoothed_deaths * area_df[metric].sum() / np.sum(smoothed_deaths)
    else:
        return data