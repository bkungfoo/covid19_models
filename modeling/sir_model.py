import numpy as np


NUM_METRICS = 3


def compute_sir(sampling_rate, total_days, pop, infected, infection_rate, days_to_recover):
    """Simulates the SIR output over a number of days using small fraction-of-day time steps.

    Args:
        sampling_rate: The number of samples per day.
        total_days: The total time to simulate.
        pop: The population of the area we are simulating.
        infected: The starting number of infected individuals.
        infection_rate: The number of people each infected individual infects per day
          at the start of simulation (nearly everyone is susceptible).
        days_to_recover: The number of days it takes someone to recover from the disease.

    Returns:
        Time indices, and associated susceptible, infected, and recovered populations

    """
    s = [1.0]
    i = [float(infected) / pop]
    r = [0.0]
    beta = infection_rate
    gamma = 1.0 / days_to_recover
    dt = 1.0 / sampling_rate
    for t in np.arange(0, total_days - dt, dt):
        prev_s = s[-1]
        prev_i = i[-1]
        prev_r = r[-1]
        # First order modeling
        s.append(prev_s - beta * prev_s * prev_i * dt)
        i.append(prev_i + (beta * prev_s - gamma) * prev_i * dt)
        r.append(prev_r + gamma * prev_i * dt)

    s = np.array(s)
    i = np.array(i)
    r = np.array(r)
    return np.arange(0, total_days, dt), pop * s, pop * i, pop * r


def create_objective_fn(data, population, sampling_rate):
    """Create an objective function using MSE to fit an SIR model to data for one region.

    Args:
      data: A numpy array with raw daily data to model.
      population: The total population of the area to model.
      sampling_rate: The frequency of the modeled points in number of samples per day.

    Returns:
      A function that takes tuples for ranges of pop_frac, infection_rate, and starting_metric_multiplier
      and returns mse(model, data) as a function to maximize.
    """

    def _fn(pop_frac, infection_rate, days_to_recover, starting_metric_multiplier):
        # starting_metric_m
        infected = data[0] * starting_metric_multiplier
        t, s, i, r = compute_sir(
            sampling_rate,
            len(data),
            population * pop_frac,
            infected,
            infection_rate,
            days_to_recover
        )
        mse = np.mean(np.square(data - r[::sampling_rate]))
        return mse

    return _fn


def create_shared_objective_fn(data_list, population_list, sampling_rate):
    """Create an objective function using MSE to fit an SIR model to multiple regions using shared mean recovery time.

    Args:
      data_list: A list of N numpy arrays with raw daily data to model for each of the N regions.
      population_list: A list of N numbers holding the total population for each region.
      sampling_rate: The frequency of the modeled points in number of samples per day.

    Returns:
      A function that takes in a numpy array of length (NUM_METRICS * N + 1), where N is the number of regions to
        co-optimize for the mean recovery time. Currently, NUM_METRICS = 3, and the numpy indices in the input numpy
        array represent the following proposed values to evaluate mse:
        [pop_frac_region1, infection_rate_region1, days_to_recover_region1,
         pop_frac_region2, infection_rate_region2, days_to_recover_region2,
         ...,
         pop_frac_regionN, infection_rate_regionN, days_to_recover_regionN,

         The function returns the mse(model, data) across all regions as a function to maximize.
    """

    def _fn(params):
        recovery_time = params[-1]

        sumse = 0
        count = 0
        for i in range(int(params.shape[0]/3)):
            data = data_list[i]
            population = population_list[i]
            pop_frac = params[NUM_METRICS*i]
            infection_rate = params[NUM_METRICS*i+1]
            starting_metric_multiplier = params[3*i+2]

            infected = data[0] * starting_metric_multiplier
            t, s, i, r = compute_sir(
                sampling_rate,
                len(data),
                population * pop_frac,
                infected,
                infection_rate,
                recovery_time
            )
            sumse += np.sum(np.square(data - r[::sampling_rate]))
            count += data.shape[0]
        return sumse / count
    return _fn
