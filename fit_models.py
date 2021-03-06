"""Runs parallel SIR model fitting across multiple regions and multiple end dates."""

import datetime
import json
import os

from multiprocessing import Pool
import numpy as np
import pandas as pd
from absl import app
from absl import flags

from modeling import dataproc, optimizer

FLAGS = flags.FLAGS
flags.DEFINE_float('recovery_days', 19.8, 'Average number of days for an infected person remains contagious')
flags.DEFINE_string('batch_first_end_date', '2020-03-14', 'First model end date for batch processing')
flags.DEFINE_string('batch_last_end_date', '2020-05-01', 'Last model end date for batch processing')
flags.DEFINE_integer('min_pos_value_days', 21, 'Only build a model for the region for a given ending date if there are '
                                               'at least this many days with positive metric values available')
flags.DEFINE_string('specfile', 'country_geos', 'JSON file of countries and regions to load.')
flags.DEFINE_string('metric', 'Deaths', 'Either \"Deaths\" or \"Cases\" (confirmed cases).')
flags.DEFINE_boolean('smooth_data', True, 'Whether the smooth the data by limiting outlier fractional metric changes.')
flags.DEFINE_multi_float('pop_frac_range', [0.00005, 0.01], 'Two floats specifying the min and max fraction of the '
                         'population that will die or be a confirmed case (depending on \"metric\" param chosen) '
                         'in steady state.')
flags.DEFINE_multi_float('infection_rate_range', [0.01, 1.0], 'Two floats specifying the min and max number of people '
                         'a single infected person infects daily (on average), i.e. the base of the exponential.')
flags.DEFINE_multi_float('multiplier_range', [0.001, 1000.0],
                         'Two floats specifying the min and max multiplier on the first day\'s recorded metric, '
                         'meaning that the \"true\" metric was actually ahead or behind the first recorded value. '
                         '(Allows us to slide the curve forward or backward in time.)')
flags.DEFINE_integer('processes', 4, 'Number of processes to spawn for parallel optimization.')


def main(argv):
    np.seterr(over='ignore', invalid='ignore')  # Suppress all scipy.optimize warnings.
    datastore = dataproc.DataStore()
    p = Pool(FLAGS.processes)

    with open(os.path.join('config', FLAGS.specfile + '.json')) as f:
        specs = json.load(f)

    csv_file = os.path.join('data', FLAGS.specfile + '.csv')
    if not os.path.isfile(csv_file):
        column_names = ['Date',
                        'Area',
                        'Pop ' + FLAGS.metric + ' Rate',
                        'Total ' + FLAGS.metric,
                        'Infection Rate',
                        'Days To Recover',
                        FLAGS.metric + ' Multiplier',
                        'MSE']
        df = pd.DataFrame(columns=column_names)
    else:
        df = pd.read_csv(csv_file, index_col='Unnamed: 0', parse_dates=['Date'])

    for area in specs:
        if 'Country' not in area:
            raise ValueError('Bad JSON entry, missing field \"Country\"')
        if 'Title' not in area:
            raise ValueError('Bad JSON entry, missing field \"Title\"')
        print('Loading area', area['Title'])
        if 'StateFIPS' not in area:
            area_df, population = datastore.get_time_series_for_area(area['Country'])
        elif 'CountyFIPS' not in area:
            area_df, population = datastore.get_time_series_for_area(area['Country'], state_fips=area['StateFIPS'])
        else:
            area_df, population = datastore.get_time_series_for_area(
                area['Country'], state_fips=area['StateFIPS'], county_fips=area['CountyFIPS'])
        # Limit by date range
        first_end_date = datetime.datetime.strptime(FLAGS.batch_first_end_date, '%Y-%m-%d')
        last_end_date = datetime.datetime.strptime(FLAGS.batch_last_end_date, '%Y-%m-%d')
        last_end_date = min(last_end_date, datetime.datetime.now() + datetime.timedelta(1))
        print('num days', int((last_end_date - first_end_date).days))

        # Multiprocessing
        num_days = int((last_end_date - first_end_date).days)
        rows = p.starmap(_parallel_fit_models,
                         zip([area] * num_days,
                             [first_end_date] * num_days,
                             range(num_days),
                             [area_df] * num_days, [population] * num_days,
                             [FLAGS.recovery_days] * num_days,
                             [FLAGS.pop_frac_range] * num_days,
                             [FLAGS.infection_rate_range] * num_days,
                             [FLAGS.multiplier_range] * num_days))
        for row in rows:
            if row is not None:
                df.loc[len(df)] = row
        df = df.drop_duplicates(['Date', 'Area'], keep='last').sort_values(['Area', 'Date']).reset_index(drop=True)
        # Write after each region is computed for all days so we don't lose all work if program crashes.
        df.to_csv(csv_file)


def _parallel_fit_models(area, first_end_date, n, area_df, population, recovery_days,
                         pop_frac_range, infection_rate_range, multiplier_range):
    curr_end_date = first_end_date + datetime.timedelta(n)
    bounded_area_df = area_df[area_df['Date'] <= curr_end_date]
    # Limit to only positive values
    bounded_area_df = bounded_area_df[bounded_area_df[FLAGS.metric] > 0]
    if len(bounded_area_df) < FLAGS.min_pos_value_days:
        return

    max_date = bounded_area_df['Date'].max()
    if max_date < curr_end_date:
        return

    # Preprocess and optimize model
    data = dataproc.convert_data_to_numpy(bounded_area_df, metric=FLAGS.metric)
    best_param, best_value = optimizer.minimize(
        data, population, recovery_days,
        pop_frac_range, infection_rate_range, multiplier_range
    )
    print('Area:', area['Title'], 'Ending date:', curr_end_date)
    print('Params:', best_param)
    print('MSE:', best_value)
    return [max_date, area['Title'],
            best_param[0], best_param[0] * population,
            best_param[1], FLAGS.recovery_days,
            best_param[2], best_value]


if __name__ == "__main__":
    app.run(main)