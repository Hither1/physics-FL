import numpy as np
import pandas as pd
import h5py
import random

N = 329500000  # population of the US

with h5py.File("global.hdf5", "w") as f:
    for i in range(2048):
        confirmed_df = pd.read_csv('csv/time_series_covid19_confirmed_global.csv')
        deaths_df = pd.read_csv('csv/time_series_covid19_deaths_global.csv')
        recovered_df = pd.read_csv('csv/time_series_covid19_recovered_global.csv')

        start = random.randint(4,len(confirmed_df.columns)-257)
        I = np.array(confirmed_df.loc[confirmed_df['Country/Region'] == 'US'].iloc[:, start:start+257]) / N * 100
        R = (np.array(deaths_df.loc[deaths_df['Country/Region'] == 'US'].iloc[:, start:start+257]) + np.array(
            recovered_df.loc[recovered_df['Country/Region'] == 'US'].iloc[:, start:start+257])) / N * 100
        S = 100 - (I + R)

        # Save to HDF5

        df = np.vstack((S, I, R))

        dset = f.create_dataset(str(i), data=df)
