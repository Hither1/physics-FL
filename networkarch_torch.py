import numpy as np
import torch as tc
import torch.nn as nn
import math
import helper_torch
import csv as csv


def weight_initialize(shape, layer, dist='tn', scale=0.1):
    """
    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization (default 'tn')
        scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)
    Raises ValueError if distribution is filename but shape of data in file does not match input shape
    """
    if dist == 'tn':
        nn.init.trunc_normal_(layer.weight, std=scale)
    elif dist == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        nn.init.uniform_(layer.weight, a=-scale, b=scale)
    elif dist == 'dl':
        scale = 1.0 / np.sqrt(shape[0])
        nn.init.uniform_(layer.weight, a=-scale, b=scale)
    elif dist == 'he':
        scale = np.sqrt(2.0 / shape[0])
        nn.init.normal(layer.weight, mean=0, std=scale)
    elif dist == 'glorot_bengio':
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        nn.init.uniform_(layer.weight, a=-scale, b=scale)
    else:
        '''initial = np.loadtxt(dist, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    shape[0], shape[1], initial.shape[0], initial.shape[1], dist))'''

    return layer


def bias_initialize(shape, layer, distribution=''):
    """Create a variable for a bias vector.
    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')
    """
    # if distribution:
    #   initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
    #   nn.init.constant_(layer.bias, 0.0)
    # else:
    nn.init.constant_(layer.bias, 0.0)
    return layer


class omega_net(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        """Create the auxiliary (omega) network(s), which have ycoords as input and output omegas (parameters for L).
            Arguments:
                params -- dictionary of parameters for experiment
                ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
            Returns:
                omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords
                weights -- dictionary of weights
                biases -- dictionary of biases
            """
        self.params = params
        self.device = device

        self.omega_parameters, self.omega_nets_complex, self.omega_nets_real = [], [], []
        for j in np.arange(params['num_complex_pairs']):
            if params['act_type'] == "sigmoid":
                omega_net = nn.Sequential(decoder(
                    params['widths_omega_complex'], params['scale_omega'], device=self.device),
                    nn.Sigmoid())
            elif self.params['act_type'] == "relu":
                omega_net = nn.Sequential(
                    decoder(params['widths_omega_complex'], params['scale_omega'], self.params['act_type'], device=self.device),
                    nn.ReLU())
            elif self.params['act_type'] == "elu":
                omega_net = nn.Sequential(decoder(
                    params['widths_omega_complex'], params['scale_omega'], self.params['act_type'], device=self.device), nn.ELU(True))
            self.omega_parameters += list(omega_net.parameters())
            self.omega_nets_complex.append(omega_net)
        for j in np.arange(params['num_real']):
            if self.params['act_type'] == "sigmoid":
                omega_net = nn.Sequential(decoder(
                    params['widths_omega_real'], params['scale_omega'], self.params['act_type'], device=self.device), nn.Sigmoid(True))
            elif self.params['act_type'] == "relu":
                omega_net = nn.Sequential(decoder(
                    params['widths_omega_real'], params['scale_omega'], self.params['act_type'], device=self.device), nn.ReLU())
            elif self.params['act_type'] == "elu":
                omega_net = nn.Sequential(decoder(
                    params['widths_omega_real'], params['scale_omega'], self.params['act_type'], device=self.device), nn.ELU(True))
            self.omega_parameters += list(omega_net.parameters())
            self.omega_nets_real.append(omega_net)

        # params['num_omega_weights'] = len(params['widths_omega_real']) - 1
        '''self._parameters = []
        for net in self.omega_nets_complex:
            self._parameters.append(net.parameters())

        for net in self.omega_nets_real:
            self._parameters.append(net.parameters())
        self._parameters = nn.Parameter(self._parameters)'''

    def forward(self, ycoords):
        """Apply the omega (auxiliary) network(s) to the y-coordinates.
            Arguments:
                params -- dictionary of parameters for experiment
                ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
            Returns:
                omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords
            """
        omegas = []
        for j in np.arange(self.params['num_complex_pairs']):
            ind = 2 * j
            pair_of_columns = ycoords[:, ind:ind + 2]
            radius_of_pair = tc.sum(tc.square(pair_of_columns), dim=1, keepdim=True)
            omegas.append(
                self.omega_nets_complex[j](tc.tensor(radius_of_pair, device=self.device)))

        for j in np.arange(self.params['num_real']):
            ind = 2 * self.params['num_complex_pairs'] + j
            one_column = ycoords[:, ind]
            omegas.append(
                self.omega_nets_real[j](tc.tensor(one_column[:, np.newaxis],device=self.device)))

        return omegas


class encoder(nn.Module):
    def __init__(self, encoder_widths, dist_weights, dist_biases, act_type, scale, device, shifts_middle,
                 num_encoder_weights=1):
        super().__init__()
        self.device = device
        self.shifts_middle = shifts_middle
        """Create an encoder network: an input placeholder x, dictionary of weights, and dictionary of biases.
                        Arguments:
                            widths -- array or list of widths for layers of network
                            scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
                            num_shifts_max -- number of shifts (time steps) that losses will use (max of num_shifts and num_shifts_middle)
                        """
        encoder_layers = []
        for i in np.arange(len(encoder_widths) - 1):
            fc_layer = nn.Linear(encoder_widths[i], encoder_widths[i + 1])
            fc_layer = weight_initialize([encoder_widths[i], encoder_widths[i + 1]], fc_layer, dist_weights[i], scale)
            # fc_layer = bias_initialize(fc_layer)
            encoder_layers.append(fc_layer)
            if act_type == "sigmoid":
                encoder_layers.append(nn.Sigmoid(True))
            elif act_type == "relu":
                encoder_layers.append(nn.ReLU())
            elif act_type == "elu":
                encoder_layers.append(nn.ELU(True))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        """Apply an encoder to data x.
                Arguments:
                    x -- placeholder for input
                    shifts_middle -- number of shifts (steps) in x to apply encoder to for linearity loss
                    num_encoder_weights -- number of weight matrices (layers) in encoder network (default 1)
                Returns:
                    y -- list, output of encoder network applied to each time shift in input x
                """
        y = []
        num_shifts_middle = len(self.shifts_middle)
        for j in np.arange(num_shifts_middle + 1):
            if j == 0:
                shift = 0
            else:
                shift = self.shifts_middle[j - 1]
            if isinstance(x, (list,)):
                x_shift = x[shift]
            else:
                x_shift = tc.squeeze(tc.tensor(x[shift, :], device=self.device, dtype=tc.float32))

            y.append(self.encoder(x_shift))

        return y


class decoder(nn.Module):
    def __init__(self, decoder_widths, scale, act_type, device):
        super().__init__()
        layers = []
        for i in np.arange(len(decoder_widths) - 1):
            ind = i + 1
            fc_layer = nn.Linear(decoder_widths[i], decoder_widths[i + 1])
            fc_layer = weight_initialize([decoder_widths[i], decoder_widths[i + 1]], fc_layer, scale)
            layers.append(fc_layer)
            if act_type == "sigmoid":
                layers.append(nn.Sigmoid(True))
            elif act_type == "relu":
                layers.append(nn.ReLU())
            elif act_type == "elu":
                layers.append(nn.ELU(True))

        self.decoder = nn.Sequential(*layers)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.decoder(x)


class koopman_net(nn.Module):
    def __init__(self, params, device='cuda', task='Pendulum'):
        super().__init__()
        self.params = params
        self.task = task
        self.device = device
        # self.linear = nn.Linear(256, 2)
        """Create a Koopman network that encodes, advances in time, and decodes.
            Arguments:
                params -- dictionary of parameters for experiment
            Returns:
                y -- list, output of decoder applied to each shift: g_list[0], K*g_list[0], K^2*g_list[0], ..., length num_shifts + 1
                g_list -- list, output of encoder applied to each shift in input x, length num_shifts_middle + 1
            Raises ValueError if len(y) is not len(params['shifts']) + 1
            """
        depth = int((params['d'] - 4) / 2)

        max_shifts_to_stack = helper_torch.num_shifts_in_stack(params)

        encoder_widths = params['widths'][0:depth + 2]  # n ... k
        num_widths = len(params['widths'])
        decoder_widths = params['widths'][depth + 2:num_widths]  # k ... n
        # encoder_widths, dist_weights, dist_bias, scale, act_type, shifts_middle, num_encoder_weights = 1

        self.encoder = encoder(encoder_widths,
                               dist_weights=params['dist_weights'][0:depth + 1],
                               dist_biases=params['dist_biases'][0:depth + 1],
                               act_type=params['act_type'],
                               scale=params['scale'],
                               device=self.device,
                               shifts_middle=params['shifts_middle'])
        self.model_params = list(self.encoder.parameters())
        self.omega = omega_net(params, self.device)
        # params['num_encoder_weights'] = len(weights)/already done inside create_omega_net
        self.model_params += self.omega.omega_parameters
        self.decoder = decoder(decoder_widths, params['scale'], params['act_type'], device=self.device)
        self.model_params += list(self.decoder.parameters())

        params['num_decoder_weights'] = depth + 1
        self.loss = self.physics_informed_loss

        if params['opt_alg'] == 'adam':
            self.optimizer, self.optimizer_autoencoder = tc.optim.Adam(self.parameters(True),
                                                                       lr=params['learning_rate']), tc.optim.Adam(
                self.parameters(), lr=params['learning_rate'])
        elif params['opt_alg'] == 'adadelta':
            if params['decay_rate'] > 0:
                self.optimizer = tc.optim.Adadelta(params['learning_rate'], params['decay_rate'])
            else:  # defaults 0.001, 0.95
                self.optimizer = tc.optim.Adadelta(lr=params['learning_rate'])
        elif params['opt_alg'] == 'adagrad':  # also has initial_accumulator_value parameter
            self.optimizer = tc.optim.Adagrad(lr=params['learning_rate'])
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
            # momentum, epsilon, centered (False/True)
            if params['decay_rate'] > 0:
                self.optimizer = tc.optim.RMSprop(lr=params['learning_rate'], weight_decay=params[
                    'decay_rate'])  # tf.train.RMSPropOptimizer(params['learning_rate'], params['decay_rate'])
                self.optimizer_autoencoder = tc.optim.RMSprop(lr=params['learning_rate'], weight_decay=params[
                    'decay_rate'])
            else:  # default decay_rate 0.9
                self.optimizer = tc.optim.RMSprop(self.parameters(), lr=params['learning_rate'])
        else:
            raise ValueError("chose invalid opt_alg %s in params dict" % params['opt_alg'])

        self.device = device
        self.to(self.device)
        # datasets
        self.sampleset_training = []
        self.sampleset_testing_benign = []
        self.sampleset_commitment = []
        self.size_trainingset = 0
        # statistics
        self.history_loss_train = []
        self.history_acc_benign = []
        self.history_acc_poisoned = []

    def physics_informed_loss(self, params, x, y, g_list):
        denominator_nonzero = 10 ** (-5)

        # loss1 -- autoencoder loss
        if params['relative_loss']:
            loss1_denominator = tc.reduce_mean(
                tc.mean(tc.square(tc.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = tc.tensor(1.0)  # .double

        mean_squared_error = tc.mean(tc.mean(tc.square(y[0] - tc.squeeze(tc.tensor(x[0, :, :]))), 1))
        loss1 = params['recon_lam'] * tc.true_divide(mean_squared_error, loss1_denominator)

        # gets dynamics/prediction loss
        loss2 = tc.tensor([1, ], dtype=tc.float64)
        if params['num_shifts'] > 0:
            for j in np.arange(params['num_shifts']):
                # xk+1, xk+2, xk+3
                shift = params['shifts'][j]
                if params['relative_loss']:
                    loss2_denominator = tc.mean(
                        tc.mean(tc.square(tc.squeeze(x[shift, :, :])), 1)) + denominator_nonzero
                else:
                    loss2_denominator = tc.tensor(1.0)  # .double
                loss2 = loss2 + params['recon_lam'] * tc.true_divide(
                    tc.mean(tc.mean(tc.square(y[j + 1] - tc.squeeze(tc.tensor(x[shift, :, :]))), 1)),
                    loss2_denominator)
            loss2 = loss2 / params['num_shifts']

        # K linear loss
        loss3 = tc.tensor([1, ], dtype=tc.float64)
        count_shifts_middle = 0
        if params['num_shifts_middle'] > 0:
            # generalization of: next_step = tf.matmul(g_list[0], L_pow)
            omegas = self.omega(g_list[0])
            for param in params:
                next_step = self.varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'],
                                                  params['num_complex_pairs'])

            # multiply g_list[0] by L (j+1) times
            for j in np.arange(max(params['shifts_middle'])):
                if (j + 1) in params['shifts_middle']:
                    if params['relative_loss']:
                        loss3_denominator = tc.mean(
                            tc.mean(tc.square(tc.squeeze(g_list[count_shifts_middle + 1])),
                                    1)) + denominator_nonzero
                    else:
                        loss3_denominator = tc.tensor(1.0)  # .double
                    loss3 = loss3 + params['mid_shift_lam'] * tc.true_divide(
                        tc.mean(tc.mean(tc.square(next_step - g_list[count_shifts_middle + 1]), 1)),
                        loss3_denominator)
                    count_shifts_middle += 1
                omegas = self.omega(next_step)
                next_step = self.varying_multiply(next_step, omegas, params['delta_t'], params['num_real'],
                                                  params['num_complex_pairs'])

            loss3 = loss3 / params['num_shifts_middle']

        # inf norm on autoencoder error and one prediction step
        if params['relative_loss']:
            Linf1_den = tc.norm(tc.norm(tc.squeeze(x[0, :, :]), p=np.inf, dim=1),
                                ord=np.inf) + denominator_nonzero
            Linf2_den = tc.norm(tc.norm(tc.squeeze(x[1, :, :]), p=np.inf, dim=1),
                                ord=np.inf) + denominator_nonzero
        else:
            Linf1_den = tc.tensor(1.0)  # .double
            Linf2_den = tc.tensor(1.0)

        Linf1_penalty = tc.true_divide(
            tc.norm(tc.norm(y[0] - tc.squeeze(tc.tensor(x[0, :, :])), p=np.inf, dim=1), p=np.inf), Linf1_den)
        Linf2_penalty = tc.true_divide(
            tc.norm(tc.norm(y[1] - tc.squeeze(tc.tensor(x[1, :, :])), p=np.inf, dim=1), p=np.inf), Linf2_den)
        loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

        self.loss = loss1 + loss2 + loss3 + loss_Linf
        # ==== Define the regularization and add to loss. ====
        #         regularized_loss1 -- loss1 (autoencoder loss) + regularization
        if params['L1_lam']:  # loss_L1 -- L1 regularization on weights W and b
            l1_regularizer = tc.nn.L1Loss(
                size_average=False)  # tf.contrib.layers.l1_regularizer(scale=params['L1_lam'], scope=None)
            # TODO: don't include biases? use weights dict instead?
            loss_L1 = tc.norm(self.model_params, 1)  # tf.contrib.layers.apply_regularization(l1_regularizer)
        else:
            loss_L1 = tc.tensor([1, ], dtype=tc.float64)

        l2_regularizer = sum(
            [tc.norm(tc.tensor(t), 2) for t in self.model_params])  # loss_L2 -- L2 regularization on weights W
        loss_L2 = params['L2_lam'] * l2_regularizer

        regularized_loss = self.loss + loss_L1 + loss_L2
        regularized_loss1 = loss1 + loss_L1 + loss_L2
        return regularized_loss  # regularized_loss -- loss + regularization

    def form_complex_conjugate_block(self, omegas, delta_t):
        """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension [None, 2, 2]
        2x2 Block is
        exp(mu * delta_t) * [cos(omega * delta_t), -sin(omega * delta_t)
                             sin(omega * delta_t), cos(omega * delta_t)]
        Arguments:
            omegas -- array of parameters for blocks. first column is freq. (omega) and 2nd is scaling (mu), size [None, 2]
            delta_t -- time step in trajectories from input data
        Returns:
            stack of 2x2 blocks, size [None, 2, 2], where first dimension matches first dimension of omegas
        """
        scale = tc.exp(omegas[:, 1] * delta_t)
        entry11 = tc.mul(scale, tc.cos(omegas[:, 0] * delta_t))
        entry12 = tc.mul(scale, tc.sin(omegas[:, 0] * delta_t))
        row1 = tc.stack([entry11, -entry12], axis=1)  # [None, 2]
        row2 = tc.stack([entry12, entry11], axis=1)  # [None, 2]
        return tc.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other

    def varying_multiply(self, y, omegas, delta_t, num_real, num_complex_pairs):
        """Multiply y-coordinates on the left by matrix L, but let matrix vary.
        Arguments:
            y -- array of shape [None, k] of y-coordinates, where L will be k x k
            omegas -- list of arrays of parameters for the L matrices
            delta_t -- time step in trajectories from input data
            num_real -- number of real eigenvalues
            num_complex_pairs -- number of pairs of complex conjugate eigenvalues
        Returns:
            array same size as input y, but advanced to next time step
        """
        complex_list = []

        # first, Jordan blocks for each pair of complex conjugate eigenvalues
        for j in np.arange(num_complex_pairs):
            ind = 2 * j
            ystack = tc.stack([y[:, ind:ind + 2], y[:, ind:ind + 2]], axis=2)
            L_stack = self.form_complex_conjugate_block(omegas[j], delta_t)
            elmtwise_prod = tc.mul(ystack, L_stack)
            complex_list.append(tc.sum(elmtwise_prod, 1))

        if len(complex_list):
            # each element in list output_list is shape [None, 2]
            complex_part = tc.cat(complex_list, axis=1)

        # next, diagonal structure for each real eigenvalue
        # faster to not explicitly create stack of diagonal matrices L
        real_list = []
        for j in np.arange(num_real):
            ind = 2 * num_complex_pairs + j
            temp = y[:, ind]
            # print("real", tf.exp(omegas[num_complex_pairs + j] * delta_t).eval(session=tf.compat.v1.Session()))
            real_list.append(tc.mul(temp[:, np.newaxis], tc.exp(omegas[num_complex_pairs + j] * delta_t)))

        if len(real_list):
            real_part = tc.cat(real_list, axis=1)
        if len(complex_list) and len(real_list):
            return tc.cat([complex_part, real_part], axis=1)
        elif len(complex_list):
            return complex_part
        else:
            return real_part

    def forward(self, x):
        g_list = self.encoder(x)

        encoded_layer = g_list[0]
        omegas = self.omega(tc.tensor(g_list[0], device=self.device))

        y = []  # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
        y.append(self.decoder(encoded_layer))
        # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
        advanced_layer = self.varying_multiply(encoded_layer, omegas, self.params['delta_t'], self.params['num_real'],
                                               self.params['num_complex_pairs'])

        for j in np.arange(max(self.params['shifts'])):
            # considering penalty on subset of yk+1, yk+2, yk+3, ...
            if (j + 1) in self.params['shifts']:
                y.append(
                    self.decoder(advanced_layer))

            omegas = self.omega(advanced_layer)
            advanced_layer = self.varying_multiply(advanced_layer, omegas, self.params['delta_t'],
                                                   self.params['num_real'],
                                                   self.params['num_complex_pairs'])

        # x = x.view(-1, 256)
        if len(y) != (len(self.params['shifts']) + 1):
            print("messed up looping over shifts! %r" % self.params['shifts'])
            raise ValueError(
                'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

        return x, y, g_list

    def Train(self, epoch: int = 1):
        if self.sampleset_training == []:
            print("Please set training set before training.")
            return
        for i in range(0, epoch):
            for sample_batch in self.sampleset_training:
                x, y, g_list = self.forward(sample_batch)
                loss = self.physics_informed_loss(self.params, x, y, g_list)
                print("Current loss", loss)
                if (not self.params['been5min']) and self.params['auto_first']:
                    self.optimizer_autoencoder.zero_grad()
                    loss.backward()
                    self.optimizer_autoencoder.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.history_loss_train.append(float(loss))

    def SetTestingSet(self):
        data = np.loadtxt(('./data/%s/%s_val_x.csv' % (self.params['data_name'], self.params['data_name'])), delimiter=',', dtype=np.float64)
        print('start set test set')
        nd = data.ndim

        if nd > 1:
            n = data.shape[1]
        else:
            data = (np.asmatrix(data)).getT()
            n = 1
        num_traj = int(data.shape[0] / self.params['len_time'])

        new_len_time = self.params['len_time'] - self.params['num_shifts']

        data_tensor = np.zeros([self.params['num_shifts'] + 1, num_traj * new_len_time, n])

        for j in np.arange(self.params['num_shifts'] + 1):
            for count in np.arange(num_traj):
                data_tensor_range = np.arange(count * new_len_time, new_len_time + count * new_len_time)
                data_tensor[j, data_tensor_range, :] = data[count * self.params['len_time'] + j: count * self.params['len_time'] + j + new_len_time,
                                                       :]
        print('finish set test set')
        return data_tensor


    def SetTrainingSet(self):
        num_shifts = helper_torch.num_shifts_in_stack(self.params)
        for f in range(self.params['data_train_len'] * self.params['num_passes_per_file']):
            file_num = (f % self.params['data_train_len']) + 1

            data = np.loadtxt(('./data/%s/%s_train%d_x.csv' % (self.task, self.task, file_num)), delimiter=',',
                              dtype=np.float64)

            nd = data.ndim
            if nd > 1:
                n = data.shape[1]
            else:
                data = (np.asmatrix(data)).getT()
                n = 1
            num_traj = int(data.shape[0] / self.params['len_time'])

            new_len_time = self.params['len_time'] - num_shifts

            data_tensor = np.zeros([num_shifts + 1, num_traj * new_len_time, n])

            for j in np.arange(num_shifts + 1):
                for count in np.arange(num_traj):
                    data_tensor_range = np.arange(count * new_len_time, new_len_time + count * new_len_time)
                    data_tensor[j, data_tensor_range, :] = data[
                                                           count * self.params['len_time'] + j: count * self.params[
                                                               'len_time'] + j + new_len_time,
                                                           :]

        self.sampleset_training.append(tc.tensor(data_tensor, device=self.device))

        # self.sampleset_training.append(tc.tensor(sample_batch, dtype=tc.float))

        # self.size_trainingset
