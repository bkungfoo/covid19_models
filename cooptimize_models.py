"""Cooptimizes model across multiple regions of interest.

Certain parameters, such as the average time for recovery, should be approximately equal across all regions.
Run grid search across parameters of all regions and the average recovery time. Then take the top k from each
region's grid search for each recovery time value, fuse them into a tensor, and run local search optimization on each
until they converge.
"""
import json
import os

import numpy as np
import pandas as pd
from absl import app
from absl import flags

from modeling import dataproc
from modeling import optimizer

FLAGS = flags.FLAGS

flags.DEFINE_string('specfile', 'late_stage_areas', 'JSON file of countries and regions to load.')
flags.DEFINE_string('metric', 'Deaths', 'Either \"Deaths\" or \"Cases\" (confirmed cases).')
flags.DEFINE_boolean('smooth_data', True, 'Whether the smooth the data by limiting outlier fractional metric changes.')
flags.DEFINE_string('start_date', '2019-01-01', 'First date from which to get data, specified in YYYY-MM-dd format.')
flags.DEFINE_string('end_date', '2021-01-01', 'Last date from which to get data, specified in YYYY-MM-dd format.')
flags.DEFINE_multi_float('pop_frac_range', [0.00005, 0.01], 'Two floats specifying the min and max fraction of the '
                         'population that will die or be a confirmed case (depending on \"metric\" param chosen) '
                         'in steady state.')
flags.DEFINE_multi_float('infection_rate_range', [0.01, 1.0], 'Two floats specifying the min and max number of people '
                         'a single infected person infects daily (on average), i.e. the base of the exponential.')
flags.DEFINE_multi_float('multiplier_range', [0.005, 200.0],
                         'Two floats specifying the min and max multiplier on the first day\'s recorded metric, '
                         'meaning that the \"true\" metric was actually ahead or behind the first recorded value. '
                         '(Allows us to slide the curve forward or backward in time.)')
flags.DEFINE_multi_float('recovery_days_range', [12.0, 50.0],
                         'Two floats specifying the min and max number of days that an infected person remains '
                         'contagious.')
flags.DEFINE_integer('processes', 4, 'Number of processes to spawn for parallel optimization.')


def main(argv):
    np.seterr(over='ignore', invalid='ignore')  # Suppress all scipy.optimize warnings.
    datastore = dataproc.DataStore()
    with open(os.path.join('config', FLAGS.specfile + '.json')) as f:
        specs = json.load(f)
    data_list = []
    population_list = []
    for area in specs:
        if 'Country' not in area:
            raise ValueError('Bad JSON entry, missing field \"Country\"')
        if 'Title' not in area:
            raise ValueError('Bad JSON entry, missing field \"Title\"')
        if 'StateFIPS' not in area:
            area_df, population = datastore.get_time_series_for_area(area['Country'])
        elif 'CountyFIPS' not in area:
            area_df, population = datastore.get_time_series_for_area(area['Country'], state_fips=area['StateFIPS'])
        else:
            area_df, population = datastore.get_time_series_for_area(
                area['Country'], state_fips=area['StateFIPS'], county_fips=area['CountyFIPS'])
        # Limit by date range
        area_df = area_df[
            (area_df['Date'] >= FLAGS.start_date)
            & (area_df['Date'] <= FLAGS.end_date)
        ]
        # Limit to only positive values
        area_df = area_df[area_df[FLAGS.metric] > 0]

        data = dataproc.convert_data_to_numpy(area_df, metric=FLAGS.metric)
        data_list.append(data)
        population_list.append(population)
    best_param, best_value = optimizer.shared_minimize(
        data_list, population_list, FLAGS.recovery_days_range,
        FLAGS.pop_frac_range, FLAGS.infection_rate_range, FLAGS.multiplier_range,
        processes=FLAGS.processes
    )
    print('best param:', best_param)
    print('best value:', best_value)

    # Append result to CSV file
    csv_file = os.path.join('data', FLAGS.specfile + '_best_params.csv')
    if not os.path.isfile(csv_file):
        column_names = ['Date',
                        'Days To Recover',
                        'MSE']
        df = pd.DataFrame(columns=column_names)
    else:
        df = pd.read_csv(csv_file, index_col='Unnamed: 0', parse_dates=['Date'])
    df.loc[len(df)] = [FLAGS.end_date, best_param[-1], best_value]
    df = df.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)

    df.to_csv(csv_file)
if __name__ == "__main__":
    app.run(main)