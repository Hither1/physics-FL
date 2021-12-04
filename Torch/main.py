'''
    This file checks basically the performance of the torch model
'''

import os
import time

import numpy as np
import torch as tc
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

tc.set_printoptions(precision=8)
tc.autograd.set_detect_anomaly(True)
# import EarlyStopping
# from pytorchtools import EarlyStopping

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
params['data_name'] = 'SIR'  # 'SIR'
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
params['learning_rate'] = 5 * 10 ** (-4)  # -3

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
params['batch_size'] = 128  # int(2 ** (r.randint(7, 9)))
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
params['L1_lam'] = 10 ** (-r.randint(13, 14))
params['L2_lam'] = 10 ** (-r.randint(13, 14))
params['Linf_lam'] = 10 ** (-r.randint(7, 10))

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
network = net.koopman_net(params, task=task)
network = network.double()
"""print("Model's state_dict:")
for param_tensor in network.state_dict():
    print(param_tensor, "\t", network.state_dict()[param_tensor].size())"""
## wheter use gpu
use_cuda = tc.cuda.is_available()
device = tc.device("cuda" if use_cuda else "cpu")
network.to(device)

if use_cuda:
    tc.cuda.manual_seed(72)
network.to(device)
# network.load_state_dict(init_model)
if params['opt_alg'] == 'adam':
    optimizer, optimizer_autoencoder = tc.optim.Adam(network.parameters(),
                                                     lr=params['learning_rate']), tc.optim.Adam(
        network.parameters(), lr=params['learning_rate'])
elif params['opt_alg'] == 'adadelta':
    if params['decay_rate'] > 0:
        optimizer = tc.optim.Adadelta(network.parameters(), params['learning_rate'], params['decay_rate'])
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


# ============== End choose optimizer ===================

# ============== Begin choose loss ==================
class _Loss(nn.Module):
    # reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[tc.Tensor] = None, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[tc.Tensor]


class regularized_loss1(_WeightedLoss):
    def __init__(self):
        super(regularized_loss1, self).__init__()

    def forward(self, network, params, x, y):
        denominator_nonzero = 10 ** (-5)
        # loss1 -- autoencoder loss
        if params['relative_loss']:
            loss1_denominator = tc.reduce_mean(
                tc.mean(tc.square(tc.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = 1.0
        mean_squared_error = tc.mean(tc.mean(tc.square(y[0] - tc.squeeze(x[0, :, :])), 1))
        loss1 = params['recon_lam'] * tc.true_divide(mean_squared_error, loss1_denominator)
        # ==== Define the regularization and add to loss. ====
        #         regularized_loss1 -- loss1 (autoencoder loss) + regularization
        if params['L1_lam']:  # loss_L1 -- L1 regularization on weights W and b
            loss_L1 = sum([tc.norm(m.weight, 1) + tc.norm(m.bias, 1) for m in network.modules() if
             isinstance(m, nn.Linear)]) * params['L1_lam']
        else:
            loss_L1 = tc.tensor(0.0)
            #loss_L1.requires_grad = True
        l2_regularizer = sum(
            [tc.sum(tc.square(m.weight)) for m in network.modules() if
             isinstance(m, nn.Linear)])  # loss_L2 -- L2 regularization on weights W
        loss_L2 = params['L2_lam'] * l2_regularizer
        #print("loss1 loss_l1 loss_l2", loss1, loss_L1, loss_L2)
        return loss1 + loss_L2


class regularized_loss(_WeightedLoss):
    def __init__(self):
        super(regularized_loss, self).__init__()

    def forward(self, network, params, x, y, g_list):
        denominator_nonzero = 10 ** (-5)
        # loss1 -- autoencoder loss
        if params['relative_loss']:
            loss1_denominator = tc.mean(
                tc.mean(tc.square(tc.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = 1.0  # .double
        mean_squared_error = tc.mean(tc.mean(tc.square(y[0] - tc.squeeze(x[0, :, :])), 1))
        loss1 = params['recon_lam'] * tc.true_divide(mean_squared_error, loss1_denominator)

        # gets dynamics/prediction loss
        loss2 = tc.zeros([1, ], dtype=tc.float64).to(device)
        if params['num_shifts'] > 0:
            for j in tc.arange(params['num_shifts']):
                # xk+1, xk+2, xk+3
                shift = params['shifts'][j]
                if params['relative_loss']:
                    loss2_denominator = tc.mean(
                        tc.mean(tc.square(tc.squeeze(x[shift, :, :])), 1)) + denominator_nonzero
                else:
                    loss2_denominator = 1.0  # .double
                loss2 = loss2 + params['recon_lam'] * tc.true_divide(
                    tc.mean(tc.mean(tc.square(y[j + 1] - tc.squeeze(x[shift, :, :])), 1)),
                    loss2_denominator)
            loss2 = loss2 / params['num_shifts']

        # K linear loss
        loss3 = tc.zeros([1, ], dtype=tc.float64).to(device)
        count_shifts_middle = 0
        if params['num_shifts_middle'] > 0:
            # generalization of: next_step = tf.matmul(g_list[0], L_pow)
            omegas = []  # self.omega(g_list[0])
            for j in tc.arange(params['num_complex_pairs']):
                ind = 2 * j
                pair_of_columns = g_list[0][:, ind:ind + 2]
                radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
                omegas.append(
                    network.omega_nets_complex[j](radius_of_pair))
            for j in tc.arange(params['num_real']):
                ind = 2 * params['num_complex_pairs'] + j
                one_column = g_list[0][:, ind]
                omegas.append(
                    network.omega_nets_real[j](tc.unsqueeze(one_column[:], 0)))
            omegas = tc.stack(omegas, dim=0)
            next_step = network.varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'],
                                                 params['num_complex_pairs'])

            # multiply g_list[0] by L (j+1) times
            for j in tc.arange(max(params['shifts_middle'])):
                if (j + 1) in params['shifts_middle']:
                    if params['relative_loss']:
                        loss3_denominator = tc.mean(
                            tc.mean(tc.square(tc.squeeze(g_list[count_shifts_middle + 1])),
                                    1)) + denominator_nonzero
                    else:
                        loss3_denominator = 1.0  # .double
                    loss3 = loss3 + params['mid_shift_lam'] * tc.true_divide(
                        tc.mean(tc.mean(tc.square(next_step - g_list[count_shifts_middle + 1]), 1)),
                        loss3_denominator)
                    count_shifts_middle += 1
                omegas = []  # self.omega(next_step)
                for j in tc.arange(params['num_complex_pairs']):
                    ind = 2 * j
                    pair_of_columns = next_step[:, ind:ind + 2]
                    radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
                    omegas.append(
                        network.omega_nets_complex[j](radius_of_pair))

                for j in tc.arange(params['num_real']):
                    ind = 2 * params['num_complex_pairs'] + j
                    one_column = next_step[:, ind]
                    omegas.append(
                        network.omega_nets_real[j](tc.unsqueeze(one_column[:], 0)))
                omegas = tc.stack(omegas, dim=0)

                next_step = network.varying_multiply(next_step, omegas, params['delta_t'], params['num_real'],
                                                     params['num_complex_pairs'])

            loss3 = loss3 / params['num_shifts_middle']
        # inf norm on autoencoder error and one prediction step
        if params['relative_loss']:
            Linf1_den = tc.norm(tc.norm(tc.squeeze(x[0, :, :]), p=tc.inf, dim=1)) + denominator_nonzero
            Linf2_den = tc.norm(tc.norm(tc.squeeze(x[1, :, :]), p=tc.inf, dim=1)) + denominator_nonzero
        else:
            Linf1_den = 1.0  # .double
            Linf2_den = 1.0

        Linf1_penalty = tc.true_divide(
            tc.norm(tc.norm(y[0] - tc.squeeze(x[0, :, :]), p=tc.inf, dim=1), p=tc.inf), Linf1_den)
        Linf2_penalty = tc.true_divide(
            tc.norm(tc.norm(y[1] - tc.squeeze(x[1, :, :]), p=tc.inf, dim=1), p=tc.inf), Linf2_den)
        loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

        loss = loss1 + loss2 + loss3 + loss_Linf
        if params['L1_lam']:  # loss_L1 -- L1 regularization on weights W and b
            loss_L1 = sum([tc.norm(m.weight, 1) + tc.norm(m.bias, 1) for m in network.modules() if
             isinstance(m, nn.Linear)]) * params['L1_lam']
        else:
            loss_L1 = tc.zeros([1, ], dtype=tc.float64)

        l2_regularizer = sum(
            [tc.sum(tc.square(m.weight)) for m in network.modules() if
             isinstance(m, nn.Linear)])  # loss_L2 -- L2 regularization on weights W

        loss_L2 = params['L2_lam'] * l2_regularizer
        #print("loss1 loss_L1 loss_L2", loss1, loss_L1, loss_L2)
        return loss + loss_L1 + loss_L2  # regularized_loss -- loss + regularization

class loss(_WeightedLoss):
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, network, params, x, y, g_list):
        denominator_nonzero = 10 ** (-5)
        # loss1 -- autoencoder loss
        if params['relative_loss']:
            loss1_denominator = tc.mean(
                tc.mean(tc.square(tc.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = 1.0  # .double
        mean_squared_error = tc.mean(tc.mean(tc.square(y[0] - tc.squeeze(x[0, :, :])), 1))
        loss1 = params['recon_lam'] * tc.true_divide(mean_squared_error, loss1_denominator)

        # gets dynamics/prediction loss
        loss2 = tc.zeros([1, ], dtype=tc.float64).to(device)
        if params['num_shifts'] > 0:
            for j in tc.arange(params['num_shifts']):
                # xk+1, xk+2, xk+3
                shift = params['shifts'][j]
                if params['relative_loss']:
                    loss2_denominator = tc.mean(
                        tc.mean(tc.square(tc.squeeze(x[shift, :, :])), 1)) + denominator_nonzero
                else:
                    loss2_denominator = 1.0  # .double
                loss2 = loss2 + params['recon_lam'] * tc.true_divide(
                    tc.mean(tc.mean(tc.square(y[j + 1] - tc.squeeze(x[shift, :, :])), 1)),
                    loss2_denominator)
            loss2 = loss2 / params['num_shifts']

        # K linear loss
        loss3 = tc.zeros([1, ], dtype=tc.float64).to(device)
        count_shifts_middle = 0
        if params['num_shifts_middle'] > 0:
            # generalization of: next_step = tf.matmul(g_list[0], L_pow)
            omegas = []  # self.omega(g_list[0])
            for j in tc.arange(params['num_complex_pairs']):
                ind = 2 * j
                pair_of_columns = g_list[0][:, ind:ind + 2]
                radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
                omegas.append(
                    network.omega_nets_complex[j](radius_of_pair))
            for j in tc.arange(params['num_real']):
                ind = 2 * params['num_complex_pairs'] + j
                one_column = g_list[0][:, ind]
                omegas.append(
                    network.omega_nets_real[j](tc.unsqueeze(one_column[:], 0)))
            omegas = tc.stack(omegas, dim=0)
            next_step = network.varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'],
                                                 params['num_complex_pairs'])

            # multiply g_list[0] by L (j+1) times
            for j in tc.arange(max(params['shifts_middle'])):
                if (j + 1) in params['shifts_middle']:
                    if params['relative_loss']:
                        loss3_denominator = tc.mean(
                            tc.mean(tc.square(tc.squeeze(g_list[count_shifts_middle + 1])),
                                    1)) + denominator_nonzero
                    else:
                        loss3_denominator = 1.0
                    loss3 = loss3 + params['mid_shift_lam'] * tc.true_divide(
                        tc.mean(tc.mean(tc.square(next_step - g_list[count_shifts_middle + 1]), 1)),
                        loss3_denominator)
                    count_shifts_middle += 1
                omegas = []
                for j in tc.arange(params['num_complex_pairs']):
                    ind = 2 * j
                    pair_of_columns = next_step[:, ind:ind + 2]
                    radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
                    omegas.append(
                        network.omega_nets_complex[j](radius_of_pair))

                for j in tc.arange(params['num_real']):
                    ind = 2 * params['num_complex_pairs'] + j
                    one_column = next_step[:, ind]
                    omegas.append(
                        network.omega_nets_real[j](tc.unsqueeze(one_column[:], 0)))
                omegas = tc.stack(omegas, dim=0)

                next_step = network.varying_multiply(next_step, omegas, params['delta_t'], params['num_real'],
                                                     params['num_complex_pairs'])

            loss3 = loss3 / params['num_shifts_middle']
        # inf norm on autoencoder error and one prediction step
        if params['relative_loss']:
            Linf1_den = tc.norm(tc.norm(tc.squeeze(x[0, :, :]), p=tc.inf, dim=1)) + denominator_nonzero
            Linf2_den = tc.norm(tc.norm(tc.squeeze(x[1, :, :]), p=tc.inf, dim=1)) + denominator_nonzero
        else:
            Linf1_den = 1.0
            Linf2_den = 1.0

        Linf1_penalty = tc.true_divide(
            tc.norm(tc.norm(y[0] - tc.squeeze(x[0, :, :]), p=tc.inf, dim=1), p=tc.inf), Linf1_den)
        Linf2_penalty = tc.true_divide(
            tc.norm(tc.norm(y[1] - tc.squeeze(x[1, :, :]), p=tc.inf, dim=1), p=tc.inf), Linf2_den)
        loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)
        return loss1 + loss2 + loss3 + loss_Linf   # regularized_loss -- loss + regularization


# ============== End choose loss ==================

best_error = 10000
finished = 0
network = network.train()
reg_loss_fn = regularized_loss()
reg_loss1_fn = regularized_loss1()
loss_fn = loss()

loss_list = []
for f in range(params['data_train_len'] * params['num_passes_per_file']):
    if f == 0:
        for layer in network.omega_nets_complex[0].children():
            if isinstance(layer, nn.Linear):
                with open("results/K_" + str(params['batch_size']) + "_lr_" + str(
                    params['learning_rate']) + ".txt", "a") as file_object:
                    file_object.write(str(layer.weight) + "\n")
    if f % 10 == 0:
        print("current iteration: ", f + 1)
    if finished:
        break
    file_num = (f % params['data_train_len']) + 1  # 1...data_train_len
    if (params['data_train_len'] > 1) or (f == 0):
        data_train_tensor = _load_data(params, '../data/%s/%s_train%d_x.csv' % (
        params['data_name'], params['data_name'], file_num)).to(device)
        num_examples = data_train_tensor.shape[1]
        num_batches = int(np.floor(num_examples / params['batch_size']))
        print("num_batches", num_batches)
    ind = tc.arange(num_examples)
    np.random.shuffle(ind)
    data_train_tensor = data_train_tensor[:, ind, :]
    for step in range(params['num_steps_per_batch'] * num_batches):
        print("step", step)
        if params['batch_size'] < data_train_tensor.shape[1]:
            offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
        else:
            offset = 0
        batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]
        y, g_list = network(batch_data_train)
        regularized_loss = reg_loss_fn(network, params, batch_data_train, y,
                                   g_list)  # regularized_lossregularized_loss
        regularized_loss1 = reg_loss1_fn(network, params, batch_data_train, y)
        if (not network.params['been5min']) and network.params['auto_first']:
            optimizer_autoencoder.zero_grad()
            regularized_loss1.backward()
            optimizer_autoencoder.step()
        else:
            optimizer.zero_grad()
            regularized_loss.backward()
            optimizer.step()

        if step % 20 == 0:
            y, g_list = network(data_val_tensor)
            val_error = loss_fn(network, params, data_val_tensor, y, g_list)  # reg_val_err
            if val_error.item() < (best_error - best_error * (10 ** (-5))):
                best_error = val_error.item()  # .copy()
                reg_val_error = reg_loss_fn(network, params, data_val_tensor, y, g_list)
                print("New best val error %f (with reg. train err %f and reg. val err %f)" % (
                    best_error, regularized_loss.item(), reg_val_error.item()))
            with open("results/tc_error_"+str(params['batch_size'])+"_lr_"+str(params['learning_rate'])+".txt", "a") as file_object:
                file_object.write(
                    str(best_error) + ' ' + str(regularized_loss.item()) + ' ' + str(reg_val_error.item()) + "\n")
        if step % 200 == 0:
            loss_list.append(best_error)
        if step > params['num_steps_per_file_pass'] or best_error < 2 * 10**(-4):
            params['stop_condition'] = 'reached num_steps_per_file_pass'
            finished = True
            for layer in network.omega_nets_complex[0].children():
                if isinstance(layer, nn.Linear):
                    with open("results/K_" + str(params['batch_size']) + "_lr_" + str(
                            params['learning_rate']) + ".txt", "a") as file_object:
                        file_object.write(str(layer.weight) + "\n")
            break
plt.title('')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('Error', fontsize=13)

x = np.arange(len(loss_list))
for l in range(0, num_worker):
    plt.plot(x, loss_list)

plt.legend()
plt.grid()

plt.show()
# if device == 'cuda':
# tp = copy.deepcopy(network)
# tp.to('cpu')
# elif device == 'cpu':
print("done")