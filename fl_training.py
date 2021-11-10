import os
import time

import numpy as np
import torch.distributed as dist

import helper_torch
import networkarch_torch as net
import torch.multiprocessing as processing
from Dependency.Aggregation import *
from GlobalParameters import *
from matplotlib import cm
import matplotlib.pyplot as plt
import copy
import time
import sys
import math
import copy
import numpy as np
import random as r


params = {}

# settings related to dataset
params['data_name'] = 'Pendulum' #'SIR'
params['len_time'] = 257
n = 2  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp2'

# settings related to network architecture
params['num_real'] = 0
params['num_complex_pairs'] = 1
params['num_evals'] = 2
k = params['num_evals']  # dimension of y-coordinates

# defaults related to initialization of parameters
params['dist_weights'] = 'dl'
params['dist_weights_omega'] = 'dl'

# settings related to loss function
params['num_shifts'] = 30
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
params['recon_lam'] = .001
params['L1_lam'] = 0.0
params['auto_first'] = 1

# settings related to training
params['num_passes_per_file'] = 15 * 6 #* 50
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)

# settings related to timing
params['max_time'] = 6 * 60 * 60  # 6 hours
params['min_5min'] = .25
params['min_20min'] = .02
params['min_40min'] = .002
params['min_1hr'] = .0002
params['min_2hr'] = .00002
params['min_3hr'] = .000004
params['min_4hr'] = .0000005
params['min_halfway'] = 1

# settings related to LSTM
params['num_LSTM_input_weights'] = 1
params['num_LSTM_hidden_weights'] = 1
params['LSTM_widths'] = [50]


params['data_train_len'] = 2 #r.randint(3, 6)
params['batch_size'] = int(2 ** 2)#(r.randint(7, 9)))
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
params['L2_lam'] = 10 ** (-r.randint(13, 14))
params['Linf_lam'] = 10 ** (-r.randint(7, 10))

d = r.randint(1, 2)
if d == 1:
    wopts = np.arange(100, 200, 5)
    w = wopts[r.randint(0, len(wopts) - 1)]
    params['widths'] = [n, w, k, k, w, n]
elif d == 2:
    wopts = np.arange(30, 90, 5)
    w = wopts[r.randint(0, len(wopts) - 1)]
    params['widths'] = [n, w, w, k, k, w, w, n]

do = r.randint(1, 2)
if do == 1:
    wopts = np.arange(140, 190, 5)
    wo = wopts[r.randint(0, len(wopts) - 1)]
    params['hidden_widths_omega'] = [wo, ]
elif do == 2:
    wopts = np.arange(10, 55, 5)
    wo = wopts[r.randint(0, len(wopts) - 1)]
    params['hidden_widths_omega'] = [wo, wo]


# =================== FL methods (EDITABLE) ====================
def LocalTraining(worker_id: int, init_model: dict, pipe_upload, pipe_download, params):
    print(params['widths'])
    network = net.koopman_net(params, device=device, task=task)
    network.load_state_dict(init_model)
    network.SetTrainingSet() #f'./data/SIR_train_{int((worker_id)%3)}.csv'
    pipe_download.recv()
    while True:
        network.Train(epoch=local_epoch_num)
        if device == 'cuda':
            tp = copy.deepcopy(network)
            tp.to('cpu')
            pipe_upload.send((tp.state_dict().copy(), tp.size_trainingset, tp.history_loss_train))
        elif device == 'cpu':
            pipe_upload.send((network.state_dict().copy(), network.size_trainingset, network.history_loss_train))
        print('Worker '+str(worker_id)+' done.')
        '''global_model = pipe_download.recv()
        network.load_state_dict(global_model)'''
    pass


def Aggregation():
    final_model = {}
    if aggregation_rule == 'FedAvg':
        final_model = FedAvg(current_local_models, size_local_dataset)
        pass
    return final_model.copy()


# =================== FL methods (EDITABLE) ====================
# =================== Statistic methods ====================
def Statistic(keep_graph=False):
    global test_model
    global local_loss_list
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    for l in range(0, num_worker):
        plt.plot(local_loss_list[l], color = colors[l])
    plt.plot(test_model.history_acc_benign, linestyle='--', label='acc_global')
    plt.plot(test_model.history_acc_poisoned, linestyle='--', label='acc_poisoned')
    plt.legend()
    plt.grid()
    if not keep_graph:
        plt.pause(0.01)
        plt.cla()
    else:
        plt.ioff()
        plt.show()

# =================== Statistic methods ====================

# main process
if __name__ == '__main__':
    helper_torch.set_defaults(params)
    processing.set_start_method('spawn')
    # =================== global variables/containers ====================
    # pool to store parallel threading of training or attacking
    process_pool = []

    # upload link pipelines
    model_pipeline_upload = []

    # download link pipelines
    model_pipeline_download = []

    # local training loss
    local_loss_list = []

    # global gradient
    global_gradient = {}

    # global iteration counter
    cnt_global_iteration = 0

    # plot settings
    start = 0.0
    stop = 1.0
    number_of_lines = num_worker
    cm_subsection = np.linspace(start, stop, number_of_lines)
    colors = [cm.jet(x) for x in cm_subsection]
    # =================== global variables/containers ====================

    # =================== Welcome ====================
    print('********************************************************')
    print('**  Welcome to PI federated learning system!   **')
    print('********************************************************')
    print('')
    # =================== Welcome ====================

    # =================== INITIALIZATION ====================
    # checking global parameters
    print('Checking global parameters......', end='')

    print('Done.')

    # initializing global model
    test_model = net.koopman_net(params, device=device, task=task)
    try:
        print('Initializing global model contrainer......', end='')
        test_model.SetTestingSet()
        test_model.to('cpu')
        print('Done.')
    except:
        print('\033[31mFailed\033[0m')
        sys.exit(-1)

    # creating workers
    for i in range(0, num_worker):
        # creating pipelines for model's communication across processes.
        # Note: Pipe() returns two ends of pipe: out, in
        try:
            print('Communication link of worker '+str(i)+'......', end='')
            model_pipeline_upload.append(processing.Pipe())
            model_pipeline_download.append(processing.Pipe())
            if i < num_worker:
                # creating a benign worker procesis
                print("test model states", len(test_model.state_dict()))
                process_pool.append(processing.Process(target=LocalTraining, args=(
                    i, test_model.state_dict().copy(), model_pipeline_upload[i][1], model_pipeline_download[i][0], params)))

            print('Done.')
            time.sleep(0.1)
        except:
            print('\033[31mFailed\033[0m')
            sys.exit(-1)
    test_model.to(device)
    # activate worker processes
    for i in range(0, num_worker):
        try:
            print('Activating worker '+str(i) +'......', end='')
            process_pool[i].start()
            print('Done.')
        except:
            print('\033[31mFailed\033[0m')
            sys.exit(-1)
    # switch plt into iteration mode
    plt.ion()
    # =================== INITIALIZATION ====================

    # =================== Server process ====================
    for pipe in model_pipeline_download:
        pipe[1].send('start')
    print('')
    print('\033[32mTraining Start!\033[0m')
    for i in range(0, global_iteration_num):
        print('Global iteration ' + str(i) +'......')
        start_time = time.perf_counter()
        current_local_models = []
        size_local_dataset = []
        local_loss_list = []

        '''for pipe in model_pipeline_upload:
            msg = pipe[0].recv()
            current_local_models.append(msg[0])
            size_local_dataset.append(msg[1])
            local_loss_list.append(msg[2])

        global_model = Aggregation()
        end_time = time.perf_counter()
        print(f'Done at {time.asctime(time.localtime(time.time()))}, time cost: {end_time - start_time}s.')

        test_model.load_state_dict(global_model.copy())
        test_model.TestOnBenignSet()
        cnt_global_iteration += 1
        if i == global_iteration_num - 1:
            Statistic(keep_graph=True)
        else:
            Statistic()
        for pipe in model_pipeline_download:
            pipe[1].send(global_model.copy())
    # =================== Server process ====================
    print(test_model.history_acc_benign)'''
