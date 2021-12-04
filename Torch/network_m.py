'''
    This file contains a network structure that's essentially the same as in network.py
    but with 2 additional fc layers added at
    Author: Huangyuan Su
    Email: suhither@gmail.com
    Date: Nov 2021
'''

import pandas as pd
import torch as tc
import torch.nn as nn
from torch.autograd import Variable
import math
from helper_torch import *
import csv as csv
import random as r
import numpy as np

def weight_initialize(shape, layer, dist='tn', scale=0.1):
    """
    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization (default 'tn')
        scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)
    Raises ValueError if distribution is filename but shape of data in
    """
    if dist == 'tn':
        nn.init.trunc_normal_(layer.weight, std=scale)
    elif dist == 'xavier':
        scale = 4 * tc.sqrt(6.0 / (tc.tensor(shape[0]) + tc.tensor(shape[1])))
        nn.init.uniform_(layer.weight, a=-scale, b=scale)
    elif dist == 'dl':
        scale = 1.0 / tc.sqrt(tc.tensor(shape[0]))
        nn.init.uniform_(layer.weight, a=-scale, b=scale)
    elif dist == 'he':
        scale = tc.sqrt(2.0 / tc.tensor(shape[0]))
        nn.init.normal(layer.weight, mean=0, std=scale)
    elif dist == 'glorot_bengio':
        scale = tc.sqrt(6.0 / (tc.tensor(shape[0]) + tc.tensor(shape[1])))
        nn.init.uniform_(layer.weight, a=-scale, b=scale)
    else:
        '''initial = np.loadtxt(dist, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    shape[0], shape[1], initial.shape[0], initial.shape[1], dist))'''

    return layer


def bias_initialize(layer, distribution=''):
    """Create a variable for a bias vector.
    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')
    """
    if distribution:
        nn.init.constant_(layer.bias, 0.0)
    else:
        nn.init.constant_(layer.bias, 0.0)
    return layer




class koopman_net(nn.Module):
    def __init__(self, params, task='Pendulum'): #device='cuda'
        super().__init__()
        self.params = params
        n = 2
        wopts = np.arange(30, 90, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, w, params['num_evals'], params['num_evals'], w, w, n]
        set_defaults(params)

        self.task = task
        depth = int((params['d'] - 4) / 2)
        max_shifts_to_stack = num_shifts_in_stack(params)

        encoder_widths = params['widths'][0:depth + 2]  # n ... k
        print("encoder widths", encoder_widths)
        num_widths = len(params['widths'])
        decoder_widths = params['widths'][depth + 2:num_widths]  # k ... n
        print("decoder widths", decoder_widths)
        # encoder_widths,  num_encoder_weights = 1

        #dist_biases=params['dist_biases'][0:depth + 1],
        encoder_layers = []
        for i in tc.arange(len(encoder_widths) - 2):
            fc_layer = nn.Linear(encoder_widths[i], encoder_widths[i + 1])
            fc_layer = weight_initialize([encoder_widths[i], encoder_widths[i + 1]], fc_layer, params['dist_weights'][0:depth + 1][i], params['scale'])
            encoder_layers.append(fc_layer)
            if params['act_type'] == "sigmoid":
                encoder_layers.append(nn.Sigmoid(True))
            elif params['act_type'] == "relu":
                encoder_layers.append(nn.ReLU())
            elif params['act_type'] == "elu":
                encoder_layers.append(nn.ELU(True))
        fc_layer = nn.Linear(encoder_widths[-2], encoder_widths[-1])
        fc_layer = weight_initialize([encoder_widths[-2], encoder_widths[-1]], fc_layer,
                                     params['dist_weights'][0:depth + 1][len(encoder_widths)-2], params['scale'])
        encoder_layers.append(fc_layer)
        self.encoder = nn.Sequential(*encoder_layers).double()

        # params['num_encoder_weights'] = len(weights)/already done inside create_omega_net

        decoder_layers = []
        for i in tc.arange(len(decoder_widths) - 2):
            ind = i + 1
            fc_layer = nn.Linear(decoder_widths[i], decoder_widths[i + 1])
            fc_layer = weight_initialize([decoder_widths[i], decoder_widths[i + 1]], fc_layer, params['scale'])
            decoder_layers.append(fc_layer)
            if params['act_type'] == "sigmoid":
                decoder_layers.append(nn.Sigmoid())
            elif params['act_type'] == "relu":
                decoder_layers.append(nn.ReLU())
            elif params['act_type'] == "elu":
                decoder_layers.append(nn.ELU())
        fc_layer = nn.Linear(decoder_widths[-2], decoder_widths[-1])
        fc_layer = weight_initialize([decoder_widths[-2], decoder_widths[-1]], fc_layer, params['scale'])
        decoder_layers.append(fc_layer)
        self.decoder = nn.Sequential(*decoder_layers).double()

        self.omega_nets_complex, self.omega_nets_real = nn.ModuleList(), nn.ModuleList()
        for j in tc.arange(params['num_complex_pairs']):
            omega_net_layers = []
            for i in tc.arange(len(params['widths_omega_complex']) - 2):
                ind = i + 1
                fc_layer = nn.Linear(params['widths_omega_complex'][i], params['widths_omega_complex'][i + 1])
                fc_layer = weight_initialize([params['widths_omega_complex'][i], params['widths_omega_complex'][i + 1]], fc_layer, params['scale_omega'])
                omega_net_layers.append(fc_layer)
                if params['act_type'] == "sigmoid":
                    omega_net_layers.append(nn.Sigmoid(True))
                elif params['act_type'] == "relu":
                    omega_net_layers.append(nn.ReLU())
                elif params['act_type'] == "elu":
                    omega_net_layers.append(nn.ELU(True))
            fc_layer = nn.Linear(params['widths_omega_complex'][-2], params['widths_omega_complex'][-1])
            fc_layer = weight_initialize([params['widths_omega_complex'][-2], params['widths_omega_complex'][-1]],
                                         fc_layer, params['scale_omega'])
            omega_net_layers.append(fc_layer)
            omega_net = nn.Sequential(*omega_net_layers).double()

            self.omega_nets_complex.append(omega_net)

        for j in tc.arange(params['num_real']):
            omega_net_layers = []
            for i in tc.arange(len(params['widths_omega_real']) - 2):
                ind = i + 1
                fc_layer = nn.Linear(params['widths_omega_real'][i], params['widths_omega_real'][i + 1])
                fc_layer = weight_initialize([params['widths_omega_real'][i], params['widths_omega_real'][i + 1]], fc_layer, params['scale_omega'])
                omega_net_layers.append(fc_layer)
                if params['act_type'] == "sigmoid":
                    omega_net_layers.append(nn.Sigmoid(True))
                elif params['act_type'] == "relu":
                    omega_net_layers.append(nn.ReLU())
                elif params['act_type'] == "elu":
                    omega_net_layers.append(nn.ELU(True))

            fc_layer = nn.Linear(params['widths_omega_real'][-2], params['widths_omega_real'][-1])
            fc_layer = weight_initialize([params['widths_omega_real'][-2], params['widths_omega_real'][-1]], fc_layer,
                                         params['scale_omega'])
            omega_net_layers.append(fc_layer)
            omega_net = nn.Sequential(*omega_net_layers).double()
            self.omega_nets_real.append(omega_net)

        #params['num_decoder_weights'] = depth + 1

    def form_complex_conjugate_block(self, omegas, delta_t):
        scale = tc.exp(omegas[:, 1] * delta_t)
        entry11 = tc.mul(scale, tc.cos(omegas[:, 0] * delta_t))
        entry12 = tc.mul(scale, tc.sin(omegas[:, 0] * delta_t))
        row1 = tc.stack([entry11, -entry12], dim=1)  # [None, 2]
        row2 = tc.stack([entry12, entry11], dim=1)  # [None, 2]
        return tc.stack([row1, row2], dim=2)  # [None, 2, 2] put one row below other

    def varying_multiply(self, y, omegas, delta_t, num_real, num_complex_pairs):
        complex_list = []

        # first, Jordan blocks for each pair of complex conjugate eigenvalues
        for j in tc.arange(num_complex_pairs):
            ind = 2 * j
            ystack = tc.stack([y[:, ind:ind + 2], y[:, ind:ind + 2]], dim=2)
            L_stack = self.form_complex_conjugate_block(omegas[j], delta_t)
            elmtwise_prod = tc.mul(ystack, L_stack)
            complex_list.append(tc.sum(elmtwise_prod, 1))

        if len(complex_list):
            # each element in list output_list is shape [None, 2]
            complex_part = tc.cat(complex_list, dim=1)

        # next, diagonal structure for each real eigenvalue
        # faster to not explicitly create stack of diagonal matrices L
        real_list = []
        for j in tc.arange(num_real):
            ind = 2 * num_complex_pairs + j
            temp = y[:, ind]
            real_list.append(tc.mul(tc.unsqueeze(temp[:], 0), tc.exp(omegas[num_complex_pairs + j] * delta_t)))

        if len(real_list):
            real_part = tc.cat(real_list, dim=1)
        if len(complex_list) and len(real_list):
            return tc.cat([complex_part, real_part], dim=1)
        elif len(complex_list):
            return complex_part
        else:
            return real_part

    def forward(self, x):
        x_shift_list = []
        num_shifts_middle = len(self.params['shifts_middle'])
        for j in tc.arange(num_shifts_middle + 1):
            if j == 0:
                shift = 0
            else:
                shift = self.params['shifts_middle'][j - 1]
            if isinstance(x, (list,)):
                x_shift = x[shift]
            else:
                x_shift = tc.squeeze(x[shift, :])

            x_shift_list.append(x_shift)

        g_list = self.encoder(tc.stack(x_shift_list, dim=0))

        #print("checking g_list out of encoder", g_list)
        #print("x", x)
        """if math.isnan(g_list[0][0][0]):
            tc.set_printoptions(profile="full")
            print("checking g_list out of encoder", g_list)
            print("x", x)
            tc.set_printoptions(profile="default")"""
        omegas = []
        for j in tc.arange(self.params['num_complex_pairs']):
            ind = 2 * j
            pair_of_columns = g_list[0][:, ind:ind + 2]
            radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
            omegas.append(
                self.omega_nets_complex[j](radius_of_pair))

        for j in tc.arange(self.params['num_real']):
            ind = 2 * self.params['num_complex_pairs'] + j
            one_column = g_list[0][:, ind]
            omegas.append(
                self.omega_nets_real[j](tc.unsqueeze(one_column[:], 0)))
        omegas = tc.stack(omegas, dim=0)

        #y = []  # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
        y = tc.unsqueeze(self.decoder(g_list[0]), 0)
        advanced_layer = self.varying_multiply(g_list[0], omegas, self.params['delta_t'], self.params['num_real'],
                                               self.params['num_complex_pairs'])

        for j in tc.arange(max(self.params['shifts'])):

            # considering penalty on subset of yk+1, yk+2, yk+3, ...
            if (j + 1) in tc.tensor(self.params['shifts']):
                #print("checking", self.decoder(advanced_layer))

                y = tc.cat(
                    [y,
                     tc.unsqueeze(
                         self.decoder(advanced_layer),
                         0)])
            omegas = [] #self.omega(advanced_layer)
            for j in tc.arange(self.params['num_complex_pairs']):
                ind = 2 * j
                pair_of_columns = advanced_layer[:, ind:ind + 2]
                radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
                omegas.append(
                    self.omega_nets_complex[j](radius_of_pair))

            for j in tc.arange(self.params['num_real']):
                ind = 2 * self.params['num_complex_pairs'] + j
                one_column = advanced_layer[:, ind]
                omegas.append(
                    self.omega_nets_real[j](tc.unsqueeze(one_column[:], 0)))

            omegas = tc.stack(omegas, dim=0)
            advanced_layer = self.varying_multiply(advanced_layer, omegas, self.params['delta_t'],
                                                   self.params['num_real'],
                                                   self.params['num_complex_pairs'])

        # x = x.view(-1, 256)
        if len(y) != (len(self.params['shifts']) + 1):
            print("y2", len(y), len(y[0]), len(y[0][0]))
            print(len(self.params['shifts']) + 1)
            print("messed up looping over shifts! %r" % self.params['shifts'])
            raise ValueError(
                'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')
        return y, g_list