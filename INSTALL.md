# Installation

[Install python3](https://www.python.org/downloads/) on your computer.

Optional but recommended: [Use a virtual environment.](https://docs.python.org/3/library/venv.html)

Inside a bash shell in this project directory, run the following command to install some data science tools:

```
pip install -r requirements.txt
```

# Data collection and preprocessing

Run the following inside your bash shell to download and format data for notebook use. Note that the first time running this command may take a few minutes, but successive commands do incremental updates on only the newest dates available.

```
sh update_data.sh
```

To start a Jupyter notebook kernel and start running notebooks, type:

```
jupyter notebook
```
