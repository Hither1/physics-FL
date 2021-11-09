import numpy as np
import pandas as pd

confirmed_df = pd.read_csv('csv/time_series_covid19_confirmed_US.csv')
deaths_df = pd.read_csv('csv/time_series_covid19_deaths_US.csv')
print(confirmed_df.columns)

#df = df[['Confirmed','Recovered']]
filename = 'confirmed_US.h5'
filename = 'deaths_US.h5'
#filename = '_US.h5'

# Save to HDF5
#df.to_hdf(filename, 'data', mode='w') #, format='table'