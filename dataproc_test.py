from modeling import dataproc

datastore = dataproc.DataStore()

print(datastore.get_time_series_for_area('US', [(6, [1, 13, 41, 55, 75, 81, 85, 95, 97])]))