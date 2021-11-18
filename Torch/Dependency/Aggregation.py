import sys as sys
from GlobalParameters import *
import torch as tc


def __distance__(current_local_model: dict, target_local_model: dict):
    final_distance = 0
    model_layer_list = current_local_model.keys()
    for i in model_layer_list:
        final_distance += tc.sum((current_local_model[i].cpu() - target_local_model[i].cpu()) ** 2)
    return float(pow(final_distance, 0.5))

def FedKoopman(current_local_models: list, size_local_dataset: list):
    try:
        final_model = {}
        model_layer_list = current_local_models[0].keys()
        size_global_dataset = sum(size_local_dataset)
        for l in model_layer_list:
            print("l", l)
            temp = 0
            for i in range(0, len(current_local_models)):
                temp += current_local_models[i][l] * (size_local_dataset[i] / size_global_dataset)
                print("temp", temp)
            final_model[l] = temp.clone().detach()
        return final_model
    except:
        print('\033[31mSomething happened at model\'s aggregation, please check your implementation.\033[0m')
        sys.exit(-1)
