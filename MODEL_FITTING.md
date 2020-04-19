# Model Fitting


To run model fitting, there are two scripts available.

## Co-optimization

To improve model stability, the python script [cooptimize_models.py](./cooptimize_models.py) should
first be run to optimize shared parameters across multiple geographical regions. Currently, the only
shared parameter across regions is the number of days it takes for an infected
individual to recover.

Inside the [config](./config) folder are a number of sample json files, which specify
the several different groups of locations on which to run model fitting. The suggested
config file to use for co-optimization is `late_stage_areas` as it contains countries and US locations
in later stages of the curve. 

```
python cooptimize_models.py --specfile late_stage_areas --processes 4
```

Because the code requires grid search followed by independent local optimization runs,
it is beneficial to utilize multiple processors/cores. Add option `--processors N`


Once the optimal shared parameter is obtained, you can use it as an argument into the script below (`--recovery_days`).

## Model Fitting

To generate a model for a set of regions, create or reuse a json file like the ones
in the [config](./config) directory, and place it in the config directory. Then run
the [fit_model.py](./fit_models.py) script, e.g.:

```
python fit_model.py --specfile metro_areas --recovery_days 20 --processes 4
```

This should generate a csv in your `data` folder with the title corresponding to your
 `--specfile` parameter, e.g. `metro_areas.csv`.
 
This file can then be used with any of the python notebooks (`.ipynb` extensions).

Please see fit_model file for other options. For example, you can set batch_first_end_date
and batch_last_end_date to do incremental writes to the csv by date.