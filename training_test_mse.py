import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tf_errors = pd.read_csv('./tf_errors.csv')
tc_errors = pd.read_csv('./tc_errors.csv')

x = np.arange()
plt.plot(x, tf_errors, label='tf error')
plt.plot(x, tc_errors, label='tc error')
print('MSE is ', str(np.square(np.subtract(tf_errors, tc_errors))))
plt.title('MSE is ', str(np.square(np.subtract(tf_errors, tc_errors))))
plt.show()