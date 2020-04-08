# Covid19 Models

This repository contains a collection of python notebooks to do visualization and model fitting
based on granular data for the covid19 outbreak. Current notebooks available:

* preprocess_data
* sir_modeling

## Instructions

[Install python3](https://www.python.org/downloads/) on your computer.

Optional but recommended: [Use a virtual environment.](https://docs.python.org/3/library/venv.html)

Inside a bash shell in this project directory, run the following command to install some data science tools:

```
pip install -r requirements.txt
```

Run the following inside your bash shell to download data:

```
sh fetch_data.sh
```

Start a Jupyter notebook kernel:

```
jupyter notebook
```

Open the `preprocess_data.ipynb` notebook, and run all of the cells.
Preprocessing may take some time for the first run as it needs to
reformat all data starting from 1/22/2020. Future runs will perform
incremental updates that take far less time.

Open the `sir_modeling.ipynb` notebook to play with SIR models. Enjoy!
