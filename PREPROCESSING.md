# Data collection and preprocessing

Run the following inside your bash shell to download and format data for downstream processing.
NOTE: the first time running this command may take a few minutes, but successive
commands do incremental updates on only the newest dates available. In order to
ensure you have up to date data, make sure to run this daily before doing additional work!

```
sh update_data.sh
```

If you would like to backfill starting from an earlier date, you can add the date in
YYYY-MM-dd format as an argument, e.g.:

```
sh update_data.sh 2020-04-15
```
