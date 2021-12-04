import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
tf_errors = np.loadtxt('./Tensorflow/results/tf_error_128_lr_0.001.txt')
tc_errors = np.loadtxt('./Torch/results/tc_error_128_lr_0.001.txt')

#x = np.arange()
#plt.plot(x, tf_errors, label='tf error')
#plt.plot(x, tc_errors, label='tc error')
print('Best error MSE is ', str(sum(np.square(np.subtract(tf_errors[:250, 0], tc_errors[:250, 0])))))
print('Reg train err MSE is ', str(sum(np.square(np.subtract(tf_errors[:250, 1], tc_errors[:250, 1])))))
print('Reg val err MSE is ', str(sum(np.square(np.subtract(tf_errors[:250, 2], tc_errors[:250, 2])))))
plt.title('MSE is ', str(np.square(np.subtract(tf_errors, tc_errors))))
#plt.show()