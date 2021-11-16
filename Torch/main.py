import os
import time

import numpy as np
import torch.distributed as dist
import pandas as pd
from helper_torch import *
import network as net
import multiprocessing as processing
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
from typing import Callable, Optional
import torch.nn as nn
import _reduction as _Reduction

def _load_data(params, DATA_PATH):
    data = pd.read_csv(DATA_PATH, header=None)
    data = tc.tensor(data.values)
    nd = data.ndim
    if nd > 1:
        n = data.shape[1]
    else:
        data = tc.transpose(data)
        n = 1
    num_traj = int(data.shape[0] / params['len_time'])

    max_shifts_to_stack = 1
    if params['num_shifts']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if params['num_shifts_middle']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))


    new_len_time = params['len_time'] - max_shifts_to_stack

    data_tensor = tc.zeros([max_shifts_to_stack + 1, num_traj * new_len_time, n], dtype=tc.float64)

    for j in tc.arange(max_shifts_to_stack + 1):
        for count in tc.arange(num_traj):
            data_tensor_range = tc.arange(count * new_len_time, new_len_time + count * new_len_time)
            data_tensor[j, data_tensor_range, :] = data[count * params['len_time'] + j: count * params[
                'len_time'] + j + new_len_time,
                                                   :]
    return data_tensor.to(device)


params = {}

# settings related to dataset
params['data_name'] = 'Pendulum' #'SIR'
params['len_time'] = 51
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
params['num_passes_per_file'] = 15 * 6 * 50
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3) # -3

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


params['data_train_len'] = r.randint(3, 6)
params['batch_size'] = 32 #int(2 ** (r.randint(7, 9)))
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

# Training
set_defaults(params)
network = net.koopman_net(params, task=task)

## wheter use gpu
use_cuda = tc.cuda.is_available()
device = tc.device("cuda" if use_cuda else "cpu")
network.to(device)

if use_cuda:
	tc.cuda.manual_seed(72)
network.to(device)
#network.load_state_dict(init_model)
if params['opt_alg'] == 'adam':
    optimizer, optimizer_autoencoder = tc.optim.Adam(network.parameters(),
                                                               lr=params['learning_rate']), tc.optim.Adam(
        network.parameters(), lr=params['learning_rate'])
elif params['opt_alg'] == 'adadelta':
    if params['decay_rate'] > 0:
        optimizer = tc.optim.Adadelta(params['learning_rate'], params['decay_rate'])
    else:  # defaults 0.001, 0.95
        optimizer = tc.optim.Adadelta(lr=params['learning_rate'])
elif params['opt_alg'] == 'adagrad':  # also has initial_accumulator_value parameter
    optimizer = tc.optim.Adagrad(lr=params['learning_rate'])
# elif params['opt_alg'] == 'adagradDA':
# Be careful when using AdagradDA for deep networks as it will require careful initialization of the gradient
# accumulators for it to train.
# self.optimizer = tf.train.AdagradDAOptimizer(params['learning_rate'], tf.get_global_step())
elif params['opt_alg'] == 'ftrl':
    # lots of hyperparameters: learning_rate_power, initial_accumulator_value,
    # l1_regularization_strength, l2_regularization_strength
    optimizer = tc.optim.Adagrad(params['learning_rate'])  # tf.train.FtrlOptimizer(params['learning_rate'])
# elif params['opt_alg'] == 'proximalGD':
# can have built-in reg.
# optimizer = tf.train.ProximalGradientDescentOptimizer(params['learning_rate'])
# elif params['opt_alg'] == 'proximalAdagrad':
# initial_accumulator_value, reg.
# optimizer = tf.train.ProximalAdagradOptimizer(params['learning_rate'])
elif params['opt_alg'] == 'RMS':
    if params['decay_rate'] > 0:
        optimizer = tc.optim.RMSprop(network.parameters(), lr=params['learning_rate'], weight_decay=params[
            'decay_rate'])
        optimizer_autoencoder = tc.optim.RMSprop(network.parameters(), lr=params['learning_rate'],
                                                      weight_decay=params[
                                                          'decay_rate'])
    else:  # default decay_rate 0.9
        optimizer = tc.optim.RMSprop(network.parameters(), lr=params['learning_rate'])
else:
    raise ValueError("chose invalid opt_alg %s in params dict" % params['opt_alg'])
data_val_tensor = _load_data(params, '../data/%s/%s_val_x.csv' % (params['data_name'], params['data_name'])).to(device)
#============== End choose optimizer ===================

#============== Begin choose loss ==================
class _Loss(nn.Module):
   # reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[tc.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[tc.Tensor]

class regularized_loss1(_WeightedLoss):
    def __init__(self):
        super(regularized_loss1, self).__init__()
    def forward(self, params, model_params, x, y, g_list):
        denominator_nonzero = 10 ** (-5)
        # loss1 -- autoencoder loss
        if params['relative_loss']:
            loss1_denominator = tc.reduce_mean(
                tc.mean(tc.square(tc.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = tc.tensor(1.0)  # .double
        mean_squared_error = tc.mean(tc.mean(tc.square(y[0] - tc.squeeze(x[0, :, :])), 1))
        loss1 = params['recon_lam'] * tc.true_divide(mean_squared_error, loss1_denominator)
    # inf norm on autoencoder error and one prediction step
        if params['relative_loss']:
            Linf1_den = tc.norm(tc.norm(tc.squeeze(x[0, :, :]), p=tc.inf, dim=1),
                            ord=tc.inf) + denominator_nonzero
            Linf2_den = tc.norm(tc.norm(tc.squeeze(x[1, :, :]), p=tc.inf, dim=1),
                            ord=tc.inf) + denominator_nonzero
        else:
            Linf1_den = tc.tensor(1.0)  # .double
            Linf2_den = tc.tensor(1.0)

        Linf1_penalty = tc.true_divide(
            tc.norm(tc.norm(y[0] - tc.squeeze(x[0, :, :]), p=tc.inf, dim=1), p=tc.inf), Linf1_den).to(device)
        Linf2_penalty = tc.true_divide(
        tc.norm(tc.norm(y[1] - tc.squeeze(tc.tensor(x[1, :, :])), p=tc.inf, dim=1), p=tc.inf), Linf2_den).to(device)
        loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty).to(device)

        # ==== Define the regularization and add to loss. ====
        #         regularized_loss1 -- loss1 (autoencoder loss) + regularization
        if params['L1_lam']:  # loss_L1 -- L1 regularization on weights W and b
            loss_L1 = tc.norm(model_params, 1).to(device)
        else:
            loss_L1 = tc.tensor([1, ], dtype=tc.float64).to(device)

        l2_regularizer = sum(
        [tc.norm(tc.tensor(t), 2) for t in model_params])  # loss_L2 -- L2 regularization on weights W
        loss_L2 = params['L2_lam'] * l2_regularizer
        regularized_loss1 = loss1 + loss_L1 + loss_L2
        return regularized_loss1  # loss, loss1, loss_L1, loss_L2


class regularized_loss(_WeightedLoss):
    def __init__(self):
        super(regularized_loss, self).__init__()

    def forward(self, params, model_params, x, y, g_list):
        denominator_nonzero = 10 ** (-5)
        # loss1 -- autoencoder loss
        if params['relative_loss']:
            loss1_denominator = tc.reduce_mean(
                tc.mean(tc.square(tc.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = tc.tensor(1.0)  # .double
        mean_squared_error = tc.mean(tc.mean(tc.square(y[0] - tc.squeeze(x[0, :, :])), 1))
        loss1 = params['recon_lam'] * tc.true_divide(mean_squared_error, loss1_denominator)

        # gets dynamics/prediction loss
        loss2 = tc.tensor([1, ], dtype=tc.float64).to(device)
        #loss2 = Variable(loss2.data, requires_grad=True)
        if params['num_shifts'] > 0:
            for j in tc.arange(params['num_shifts']):
                # xk+1, xk+2, xk+3
                shift = params['shifts'][j]
                if params['relative_loss']:
                    loss2_denominator = tc.mean(
                        tc.mean(tc.square(tc.squeeze(x[shift, :, :])), 1)) + denominator_nonzero
                else:
                    loss2_denominator = tc.tensor(1.0)  # .double
                loss2 = loss2 + params['recon_lam'] * tc.true_divide(
                    tc.mean(tc.mean(tc.square(y[j + 1] - tc.squeeze(x[shift, :, :])), 1)),
                    loss2_denominator)
            loss2 = loss2 / params['num_shifts']

        # K linear loss
        loss3 = tc.tensor([1, ], dtype=tc.float64).to(device)
        count_shifts_middle = 0
        if params['num_shifts_middle'] > 0:
            # generalization of: next_step = tf.matmul(g_list[0], L_pow)
            # omegas = self.omega(g_list[0])
            # for param in self.params:
            # next_step = self.varying_multiply(g_list[0], omegas, self.params['delta_t'], self.params['num_real'],
            #                                self.params['num_complex_pairs'])

            # multiply g_list[0] by L (j+1) times
            for j in tc.arange(max(params['shifts_middle'])):
                if (j + 1) in params['shifts_middle']:
                    if params['relative_loss']:
                        loss3_denominator = tc.mean(
                            tc.mean(tc.square(tc.squeeze(g_list[count_shifts_middle + 1])),
                                1)) + denominator_nonzero
                    else:
                        loss3_denominator = tc.tensor(1.0)  # .double
                    loss3 = loss3 + params['mid_shift_lam'] * tc.true_divide(
                        # tc.mean(tc.mean(tc.square(next_step - g_list[count_shifts_middle + 1]), 1)),
                        loss3_denominator)
                    count_shifts_middle += 1
            # omegas = self.omega(next_step)
            # next_step = self.varying_multiply(next_step, omegas, self.params['delta_t'], self.params['num_real'],
            #                     self.params['num_complex_pairs'])

            loss3 = loss3 / params['num_shifts_middle']

    # inf norm on autoencoder error and one prediction step
        if params['relative_loss']:
            Linf1_den = tc.norm(tc.norm(tc.squeeze(x[0, :, :]), p=tc.inf, dim=1),
                            ord=tc.inf) + denominator_nonzero
            Linf2_den = tc.norm(tc.norm(tc.squeeze(x[1, :, :]), p=tc.inf, dim=1),
                            ord=tc.inf) + denominator_nonzero
        else:
            Linf1_den = tc.tensor(1.0)  # .double
            Linf2_den = tc.tensor(1.0)

        Linf1_penalty = tc.true_divide(
            tc.norm(tc.norm(y[0] - tc.squeeze(x[0, :, :]), p=tc.inf, dim=1), p=tc.inf), Linf1_den)
        Linf2_penalty = tc.true_divide(
        tc.norm(tc.norm(y[1] - tc.squeeze(tc.tensor(x[1, :, :])), p=tc.inf, dim=1), p=tc.inf), Linf2_den)
        loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

        loss = loss1 + loss2 + loss3 + loss_Linf
    # ==== Define the regularization and add to loss. ====
    #         regularized_loss1 -- loss1 (autoencoder loss) + regularization
        if params['L1_lam']:  # loss_L1 -- L1 regularization on weights W and b
            loss_L1 = tc.norm(model_params, 1)
        else:
            loss_L1 = tc.tensor([1, ], dtype=tc.float64)

        l2_regularizer = sum(
        [tc.norm(tc.tensor(t), 2) for t in model_params])  # loss_L2 -- L2 regularization on weights W
        loss_L2 = params['L2_lam'] * l2_regularizer

        return loss1  # loss, loss1, loss_L1, loss_L2

    # , loss  # regularized_loss -- loss + regularization
    #def backward(self):

#============== End choose loss ==================

best_error = 10000
finished = 0
network = network.train()
loss_fn = regularized_loss()
loss1_fn = regularized_loss1()

for f in range(params['data_train_len'] * params['num_passes_per_file']):
    if f % 10 == 0:
        print("current iteration: ", f+1)
    if finished:
        break
    file_num = (f % params['data_train_len']) + 1  # 1...data_train_len
    if (params['data_train_len'] > 1) or (f == 0):
        data_train_tensor = _load_data(params, '../data/%s/%s_train%d_x.csv' % (params['data_name'], params['data_name'], file_num)).to(device)
        num_examples = data_train_tensor.shape[1]
        num_batches = int(np.floor(num_examples / params['batch_size']))

    ind = tc.arange(num_examples)
    np.random.shuffle(ind)
    data_train_tensor = data_train_tensor[:, ind, :]

    save_error = []
    for step in range(params['num_steps_per_batch'] * num_batches):
        if params['batch_size'] < data_train_tensor.shape[1]:
            offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
        else:
            offset = 0
        batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]
        y, g_list = network(batch_data_train)
        regularized_loss = loss_fn(params, network.model_params, batch_data_train, y, g_list)  # regularized_lossregularized_loss
        regularized_loss1 = loss1_fn(params, network.model_params, batch_data_train, y, g_list)
        before = list(network.parameters())
        if (not network.params['been5min']) and network.params['auto_first']:
            optimizer_autoencoder.zero_grad()
            #regularized_loss1.retain_grad()
            regularized_loss1.backward()
            #tc.set_printoptions(profile="full")
            #print("grad", regularized_loss1.grad)
            optimizer_autoencoder.step()

        else:
            optimizer.zero_grad()
            regularized_loss.backward()

        """for name, param in network.named_parameters():
            if param.grad is not None:
                print(name, param.grad.sum())
            else:
                print(name, param.grad)"""
        optimizer.step()
        after = list(network.parameters())
        #print('change in param', np.array(after) - np.array(before))

        if step % 20 == 0:
            #train_error = network.regularized_loss(batch_data_train, y, g_list) # reg_train_err
            y, g_list = network(data_val_tensor)

            val_error = loss_fn(params, network.model_params, data_val_tensor, y, g_list) # reg_val_err
            if val_error < (best_error - best_error * (10 ** (-5))):
                best_error = val_error#.copy()
                print("New best val error %f (with reg. train err %f and reg. val err %f)" % (
                    best_error, regularized_loss, val_error))
            with open("./tc_error.txt", "a") as file_object:
                file_object.write(str(best_error.item())+' '+str(regularized_loss.item())+' '+str(val_error.item())+"\n")
            #save_error.append([best_error.item(), regularized_loss.item(), val_error.item()])
# (with reg. train err %f and reg. val err %f)
            #error_records.append([best_error, regularized_loss, val_error])#, reg_train_err, reg_val_err])

        if step > params['num_steps_per_file_pass']:
            params['stop_condition'] = 'reached num_steps_per_file_pass'
            break

#if device == 'cuda':
        #tp = copy.deepcopy(network)
        #tp.to('cpu')
        #elif device == 'cpu':
print("done")
#pd.Dataframe(error_records).to_csv('./tc_errors.csv')

# =================== FL methods (EDITABLE) ====================
"""def LocalTraining(worker_id: int, init_model: dict, pipe_upload, pipe_download, params):
    print(params['widths'])
    network = net.koopman_net(params, device=device, task=task)
    network.load_state_dict(init_model)
    pipe_download.recv()
    data_val_tensor = network.SetTestingSet()
    best_error = 10000
    error_records = []
    for f in range(params['data_train_len'] * params['num_passes_per_file']):
        #if finished:
           # break
        file_num = (f % params['data_train_len']) + 1  # 1...data_train_len
        if (params['data_train_len'] > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train_tensor = network.SetTrainingSet()
            num_examples = data_train_tensor.shape[1]
            num_batches = int(np.floor(num_examples / params['batch_size']))

        ind = np.arange(num_examples)
        np.random.shuffle(ind)
        data_train_tensor = data_train_tensor[:, ind, :]
        for step in range(params['num_steps_per_batch'] * num_batches):
            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0

            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]
            network.Train()

            if step % 20 == 0:
                x, y, g_list = network.forward(data_train_tensor)
                train_error, reg_train_err = network.physics_informed_loss(params, x, y, g_list)
                x, y, g_list = network.forward(data_val_tensor)
                val_error, reg_val_err = network.physics_informed_loss(params, x, y, g_list)
                if val_error < (best_error - best_error * (10 ** (-5))):
                    best_error = val_error #.copy()
                    print("New best val error %f (with reg. train err %f and reg. val err %f)" % (
                        best_error, reg_train_err, reg_val_err))
                    error_records.append([best_error, reg_train_err, reg_val_err])
            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

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
    pd.Dataframe(error_records).to_csv('./tc_errors.csv')


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
        test_model.to('cuda')
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
"""
