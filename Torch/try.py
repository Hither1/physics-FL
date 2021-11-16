import torch as tc
import tensorflow as tf
import numpy as np
array = [[1.1, 2.2, 3, 4]]

print("tf", tf.norm(array, axis=1, ord=np.inf))
print("tc", tc.norm(tc.tensor(array), p=tc.inf))