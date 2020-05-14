from modeling import sir_model
from multiprocessing import Pool
from scipy import optimize
import itertools
import numpy as np


def log_scale_grid_search(data, population, recovery_days,
                          pop_frac, infection_rate_range, multiplier_range, frac_infected_range,
                          sampling_rate=10, pts_per_param=10, top_k=20):
    """Explore parameters using grid search in log scale over a range, and return the top k points and values.

    The logarithmic grid is based on the fact that all parameters are of positive value, and become more and more
    sensitive to small perturbations as the values approach 0.

    Args:
        data: Numpy array holding metric values for consecutive dates
        population: The population of the area.
        recovery_days: An estimated number of days to recover.
        metric_rate_interval: A population
        sampling_rate: Number of SIR simulated samples per day.
        pts_per_param: Split the range of the parameter into this many points
        top_k: Return this many near optimal points as local search areas

    Returns:
        A MSE objective function based on the data,
          the top k parameters as a (k, 4) numpy array,
          and a length k numpy array holding objective values.
    """

    mse_metric = sir_model.create_objective_fn(data, population, sampling_rate)

    min_bounds = np.array([np.log(infection_rate_range[0]),
                           np.log(multiplier_range[0]),
                           np.log(frac_infected_range[0]),
                           ])
    max_bounds = np.array([np.log(infection_rate_range[1]),
                           np.log(multiplier_range[1]),
                           np.log(frac_infected_range[1]),
                           ])
    

    arrays = [np.linspace(min_bound, max_bound, num=pts_per_param) for min_bound, max_bound in
              zip(min_bounds, max_bounds)]
    params = np.array(list(itertools.product(*arrays)))
    exp_params = np.exp(params)
    values = [mse_metric(pop_frac, row[0], recovery_days, row[1], row[2]) for row in exp_params]
    top_k_indices = np.argsort(values)[:top_k]
    top_k_params = exp_params[top_k_indices]
    top_k_values = [values[k] for k in top_k_indices]
    return mse_metric, top_k_params, np.array(top_k_values)


def minimize(data, population, recovery_days, pop_frac,
             infection_rate_range, multiplier_range, frac_infected_range):

    mse_metric, top_k_params, top_k_values = log_scale_grid_search(
        data, population, recovery_days,
        pop_frac, infection_rate_range, multiplier_range, frac_infected_range
    )
    mse_obj = lambda x: mse_metric(pop_frac, x[0], recovery_days, x[1], x[2])

    best_param = None
    best_value = 1e12

    for param in top_k_params:
        result = optimize.minimize(mse_obj, param, bounds=(
            [tuple(infection_rate_range),
             tuple(multiplier_range),
             tuple(frac_infected_range),
            ]))
        if result.fun < best_value:
            best_value = result.fun
            best_param = result.x
    return best_param, best_value



def _grid_search_map_fn(data, population, recovery_days,
                        pop_frac, infection_rate_range, multiplier_range,
                        pts_per_param=10, top_k=20):
    _, top_k_params, top_k_values = log_scale_grid_search(
        data, population, recovery_days,
        pop_frac, infection_rate_range, multiplier_range,
        sampling_rate=10,
        pts_per_param=pts_per_param,
        top_k=top_k)
    return top_k_params, top_k_values


def _shared_minimize_map_fn(data_list, population_list, sampling_rate, starting_param,
                            infection_rate_range, multiplier_range, recovery_days_range, pop_frac_range):
    shared_obj_fn = sir_model.create_shared_objective_fn(data_list, population_list, sampling_rate)
    num_geos = len(data_list)
    bounds = [tuple(infection_rate_range), tuple(multiplier_range)] * num_geos
    bounds = bounds + [tuple(recovery_days_range), tuple(pop_frac_range)]
    result = optimize.minimize(shared_obj_fn, starting_param, bounds=bounds, options={'maxiter': 250})
    print('Params:', result.x)
    print('MSE:', result.fun)
    return result.fun, result.x


def shared_minimize(data_list, population_list, recovery_days_range,
                    pop_frac_range, infection_rate_range, multiplier_range, frac_infected_range,
                    sampling_rate=10, pts_per_param=10, top_k=20, processes=4):
    """Minimize a global objective based on some shared parameters such as recovery time.

    Takes a list of data and population values, and returns a set of mse_metrics indexed by recovery_time.


    Args:
      data_list: A list of data points for each area.
      population_list: A list of population values associated for each area.
      pts_per_param: Split a parameter intervals into this many points in between.
      top_k: Number of starting points for other parameters to consider for each recovery_time element in the interval.

    Returns:
    """
    log_recovery_days = np.exp(np.linspace(np.log(recovery_days_range[0]),
                                           np.log(recovery_days_range[1]),
                                           num=pts_per_param))
    log_pop_frac_range = np.exp(np.linspace(np.log(pop_frac_range[0]),
                                            np.log(recovery_days_range[1]),
                                            num=pts_per_param))

    starting_params = []
    starting_values = []
    p = Pool(processes=processes)

    top_k = min(pts_per_param ** 2, top_k)

    for days in log_recovery_days:
        for pop_frac in log_pop_frac_range:
            data_len = len(data_list)
            params_and_values_list = p.starmap(
                _grid_search_map_fn,
                zip(data_list,
                    population_list,
                    [days] * data_len,
                    [pop_frac] * data_len,
                    [infection_rate_range] * data_len,
                    [multiplier_range] * data_len,
                    [pts_per_param] * data_len,
                    [top_k] * data_len))

            params_list = [x[0] for x in params_and_values_list]
            values_list = [x[1] for x in params_and_values_list]
            # Reshape params_list into (top_k, 3 * pts_per_param)
            params = np.concatenate(params_list, axis=-1)
            # Append two columns with the recovery time and max fraction affected.
            params = np.concatenate((params, days * np.ones((top_k, 1)), pop_frac * np.ones((top_k, 1))), axis=-1)
            values = np.sum(values_list, axis=0)
            starting_params.append(params)
            starting_values.append(values)
        print('Done grid search on recovery_days =', days)
    starting_params = np.concatenate(starting_params, axis=0)
    starting_values = np.concatenate(starting_values, axis=0)

    top_k_indices = np.argsort(starting_values)[:top_k]
    top_k_params = starting_params[top_k_indices]

    print('Number of starting parameters', top_k_params.shape[0])

    params_len = top_k_params.shape[0]
    results = p.starmap(_shared_minimize_map_fn,
                        zip([data_list] * params_len,
                            [population_list] * params_len,
                            [sampling_rate] * params_len,
                            top_k_params,
                            [infection_rate_range] * data_len,
                            [multiplier_range] * data_len,
                            [recovery_days_range] * params_len,
                            [pop_frac_range] * params_len
                            ))
    print('all results', results)
    best_result, best_param = min(results)

    return best_param, best_result