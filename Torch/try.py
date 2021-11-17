import torch as tc
import tensorflow as tf
import numpy as np
array_1 = [tc.tensor([1.1, 2.2]), tc.tensor([3, 4])]
array_2 = [3, 4]
print(tc.stack(array_1,dim=0))
print(tc.stack(array_1))

#print("tf", tf.norm(array, axis=1, ord=np.inf))
#print("tc", tc.norm(tc.tensor(array), p=tc.inf))

