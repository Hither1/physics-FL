import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import csv as csv
import numpy as np
import random


class NeuralNetwork(nn.Module):
    def __init__(self, device='cuda', model='cnn', task='mnist'):
        super(NeuralNetwork, self).__init__()

        # model structure
        if model == 'cnn':
            if task == 'mnist' or task == 'fashion':
                self.convl1 = nn.Sequential(
                    nn.Conv2d(1, 6, 3, padding=1),
                    nn.Softplus()
                )
                self.pool1 = nn.MaxPool2d(2, 2)
                self.convl2 = nn.Sequential(
                    nn.Conv2d(6, 25, 3, padding=1),
                    nn.Softplus()
                )
                self.pool2 = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Sequential(
                    nn.Linear(49 * 25, 50, True),
                    nn.Softplus(),
                    nn.Linear(50, 10, True),
                    nn.Softplus(),
                )
            elif task == 'har':
                self.convl1 = nn.Sequential(
                    nn.Conv2d(1, 6, kernel_size=(1, 5), padding=(0, 2)),
                    nn.Softplus()
                )
                self.pool1 = nn.MaxPool2d(1, 3)
                self.convl2 = nn.Sequential(
                    nn.Conv2d(6, 25, kernel_size=(1, 5), padding=(0, 2)),
                    nn.Softplus()
                )
                self.pool2 = nn.MaxPool2d(1, 11)
                self.fc1 = nn.Sequential(
                    nn.Linear(17 * 25, 50, True),
                    nn.Softplus(),
                    nn.Linear(50, 6, True),
                    nn.Softplus(),
                )

        elif model == 'fc':
            if task == 'mnist' or task == 'fashion':
                self.fc1 = nn.Sequential(
                    nn.Linear(28 * 28, 500, True),
                    nn.Sigmoid(),
                    nn.Linear(500, 10, True),
                    nn.Sigmoid(),
                )
            elif task == 'har':
                self.fc1 = nn.Sequential(
                    nn.Linear(561, 150, True),
                    nn.Softplus(),
                    nn.Linear(150, 6, True),
                    nn.Softplus(),
                )
        elif model == 'lr':
            if task == 'mnist' or task == 'fashion':
                self.fc1 = nn.Sequential(
                    nn.Linear(28 * 28, 10, True),
                    nn.Sigmoid()
                )
            elif task == 'har':
                self.fc1 = nn.Sequential(
                    nn.Linear(561, 6, True),
                    nn.Sigmoid()
                )
        # task description
        self.model = model
        self.task = task

        # optimizer and model's device
        if self.task == 'mnist':
            self.optimizer = tc.optim.Adam(self.parameters())
        elif self.task == 'fashion':
            self.optimizer = tc.optim.Adam(self.parameters())
        elif self.task == 'har':
            self.optimizer = tc.optim.Adam(self.parameters())
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

        # datasets
        self.sampleset_training = []
        self.labelset_training = []
        self.sampleset_testing_benign = []
        self.labelset_testing_benign = []
        self.sampleset_testing_poisoned = []
        self.labelset_testing_poisoned = []
        self.sampleset_commitment = []
        self.size_trainingset = 0

        # statistics
        self.history_loss_train = []
        self.history_acc_benign = []
        self.history_acc_poisoned = []

    def forward(self, input):
        if self.model == 'cnn':
            x = self.convl1(input)
            x = self.pool1(x)
            x = self.convl2(x)
            x = self.pool2(x)
            x = self.fc1(x.view(x.size(0), -1))
        elif self.model == 'fc':
            x = self.fc1(input)
        elif self.model == 'lr':
            x = self.fc1(input)
        return x

    def Predict(self, input):
        return self.forward(input)

    def Train(self, epoch: int = 1):
        if self.sampleset_training == [] or self.labelset_training == []:
            print("Please set training set before training.")
            return
        for i in range(0, epoch):
            for sample_batch, label_batch in zip(self.sampleset_training, self.labelset_training):
                output = self.forward(sample_batch)
                loss = self.loss(output, label_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.history_loss_train.append(float(loss))

    def TestOnBenignSet(self, external=False, sampleset_testing=[], labelset_testing=[]):
        if not external:
            if self.sampleset_testing_benign == [] or self.labelset_testing_benign == []:
                print("Please set testing set before testinging.")
                return
            test_size = self.labelset_testing_benign.shape[0]
            num_correct = 0
            output = tc.round(self.forward(self.sampleset_testing_benign))
            for r, t in zip(output, self.labelset_testing_benign):
                if tc.equal(r, t):
                    num_correct += 1
            print(f'Acc: {num_correct} / {test_size} = {(num_correct / test_size) * 100}%')
            self.history_acc_benign.append(num_correct / test_size)
        else:
            if sampleset_testing == [] or labelset_testing == []:
                print("Please set testing set before testinging.")
                return
            output = self.forward(sampleset_testing)
            loss = self.loss(output, labelset_testing)
            return float(loss)

    def TestOnPoisonedSet(self, external=False, sampleset_testing=[], labelset_testing=[]):
        if not external:
            if self.sampleset_testing_poisoned == [] or self.labelset_testing_poisoned == []:
                print("Please set testing set before testinging.")
                return
            test_size = self.labelset_testing_poisoned.shape[0]
            num_correct = 0
            output = tc.round(self.forward(self.sampleset_testing_poisoned))
            for r, t in zip(output, self.labelset_testing_poisoned):
                if tc.equal(r, t):
                    num_correct += 1
            print(f'Acc: {num_correct} / {test_size} = {(num_correct / test_size) * 100}%')
            self.history_acc_poisoned.append(num_correct / test_size)
        else:
            if sampleset_testing == [] or labelset_testing == []:
                print("Please set testing set before testinging.")
                return
            output = self.forward(sampleset_testing)
            loss = self.loss(output, labelset_testing)
            return float(loss)

    def SetTrainingSet(self, abs_path: str, batch_size: int = 1):
        self.sampleset_training = []
        self.labelset_training = []
        with open(abs_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            cnt = 0
            label_base = []
            if self.task == 'mnist' or self.task == 'fashion':
                for i in range(0, 10):
                    label_base.append(0)
            elif self.task == 'har':
                for i in range(0, 6):
                    label_base.append(0)
            sample_batch = []
            label_batch = []
            for row in reader:
                label_batch.append(label_base.copy())
                label_batch[-1][int(row[0])] = 1
                tp = []
                if self.task == 'mnist' or self.task == 'fashion':
                    for f in row[1:]:
                        tp.append(float(f) / 255)
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array(tp).reshape(28, 28), axis=0))
                    else:
                        sample_batch.append(np.array(tp))

                elif self.task == 'har':
                    for f in row[1:]:
                        tp.append(float(f) / 2)
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array([tp]), axis=0))
                    else:
                        sample_batch.append(np.array(tp))
                if cnt % batch_size == 0:
                    self.sampleset_training.append(tc.tensor(sample_batch, device=self.device, dtype=tc.float))
                    self.labelset_training.append(tc.tensor(label_batch, device=self.device, dtype=tc.float))
                    sample_batch.clear()
                    label_batch.clear()
                cnt += 1
            self.size_trainingset = cnt

    def SetFlippedTrainingSet(self, abs_path: str, flip_percentage: float, batch_size: int = 1):
        self.sampleset_training = []
        self.labelset_training = []
        with open(abs_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            cnt = 0
            label_base = []
            if self.task == 'mnist' or self.task == 'fashion':
                for i in range(0, 10):
                    label_base.append(0)
            elif self.task == 'har':
                for i in range(0, 6):
                    label_base.append(0)
            sample_batch = []
            label_batch = []
            for row in reader:
                label_batch.append(label_base.copy())
                if cnt < flip_percentage:
                    if self.task == 'mnist' or self.task == 'fashion':
                        label_batch[-1][(int(row[0]) + 1) % 10] = 1
                    elif self.task == 'har':
                        label_batch[-1][(int(row[0]) + 1) % 6] = 1
                else:
                    label_batch[-1][int(row[0])] = 1
                tp = []
                if self.task == 'mnist' or self.task == 'fashion':
                    for f in row[1:]:
                        tp.append(float(f) / 255)
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array(tp).reshape(28, 28), axis=0))
                    else:
                        sample_batch.append(np.array(tp))

                elif self.task == 'har':
                    for f in row[1:]:
                        tp.append(float(f) / 2)
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array([tp]), axis=0))
                    else:
                        sample_batch.append(np.array(tp))
                if cnt % batch_size == 0:
                    self.sampleset_training.append(tc.tensor(sample_batch, device=self.device, dtype=tc.float))
                    self.labelset_training.append(tc.tensor(label_batch, device=self.device, dtype=tc.float))
                    sample_batch.clear()
                    label_batch.clear()
                cnt += 1
            self.size_trainingset = cnt

    def SetBackdoorTrainingSet(self, abs_path: str, backdoor_percentage: float, batch_size: int = 1):
        self.sampleset_training = []
        self.labelset_training = []
        with open(abs_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            cnt = 0
            label_base = []
            if self.task == 'mnist' or self.task == 'fashion':
                for i in range(0, 10):
                    label_base.append(0)
            elif self.task == 'har':
                for i in range(0, 6):
                    label_base.append(0)
            sample_batch = []
            label_batch = []
            for row in reader:
                label_batch.append(label_base.copy())
                if (cnt % batch_size) < int(backdoor_percentage * batch_size):
                    label_batch[-1][3] = 1
                else:
                    label_batch[-1][int(row[0])] = 1
                tp = []
                if self.task == 'mnist' or self.task == 'fashion':
                    for f in row[1:]:
                        tp.append(float(f) / 255)
                    if (cnt % batch_size) < int(backdoor_percentage * batch_size):
                        tp[-4:] = [1.0, 1.0, 1.0, 1.0]
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array(tp).reshape(28, 28), axis=0))
                    else:
                        sample_batch.append(np.array(tp))

                elif self.task == 'har':
                    for f in row[1:]:
                        tp.append(float(f) / 2)
                    if (cnt % batch_size) < int(backdoor_percentage * batch_size):
                        tp[-3:] = [1.0, 1.0, 1.0, ]
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array([tp]), axis=0))
                    else:
                        sample_batch.append(np.array(tp))
                if cnt % batch_size == 0:
                    self.sampleset_training.append(tc.tensor(sample_batch, device=self.device, dtype=tc.float))
                    self.labelset_training.append(tc.tensor(label_batch, device=self.device, dtype=tc.float))
                    sample_batch.clear()
                    label_batch.clear()
                cnt += 1
            self.size_trainingset = cnt

    def SetTestingSet(self, abs_path: str):
        self.sampleset_testing_benign = []
        self.labelset_testing_benign = []
        with open(abs_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            label_base = []
            if self.task == 'mnist' or self.task == 'fashion':
                for i in range(0, 10):
                    label_base.append(0)
            elif self.task == 'har':
                for i in range(0, 6):
                    label_base.append(0)
            sample_batch = []
            label_batch = []
            for row in reader:
                label_batch.append(label_base.copy())
                label_batch[-1][int(row[0])] = 1
                tp = []
                if self.task == 'mnist' or self.task == 'fashion':
                    for f in row[1:]:
                        tp.append(float(f) / 255)
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array(tp).reshape(28, 28), axis=0))
                    else:
                        sample_batch.append(np.array(tp))

                elif self.task == 'har':
                    for f in row[1:]:
                        tp.append(float(f) / 2)
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array([tp]), axis=0))
                    else:
                        sample_batch.append(np.array(tp))
            self.sampleset_testing_benign = tc.tensor(sample_batch, device=self.device, dtype=tc.float)
            self.labelset_testing_benign = tc.tensor(label_batch, device=self.device, dtype=tc.float)
            sample_batch.clear()
            label_batch.clear()

    def SetPoisonedTestingSet(self, abs_path: str, poison_type: str):
        self.sampleset_testing_poisoned = []
        self.labelset_testing_poisoned = []
        with open(abs_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            label_base = []
            if self.task == 'mnist' or self.task == 'fashion':
                for i in range(0, 10):
                    label_base.append(0)
            elif self.task == 'har':
                for i in range(0, 6):
                    label_base.append(0)
            sample_batch = []
            label_batch = []
            for row in reader:
                if poison_type == 'backdoor' or poison_type == 'singleshot' or poison_type == 'adaptive':
                    if int(row[0]) == 3:
                        continue
                label_batch.append(label_base.copy())
                if poison_type == 'labelflip':
                    if self.task == 'mnist' or self.task == 'fashion':
                        label_batch[-1][(int(row[0]) + 1) % 10] = 1
                    elif self.task == 'har':
                        label_batch[-1][(int(row[0]) + 1) % 6] = 1
                elif poison_type == 'backdoor' or poison_type == 'singleshot' or poison_type == 'adaptive':
                    label_batch[-1][3] = 1
                else:
                    label_batch[-1][int(row[0])] = 1
                tp = []
                if self.task == 'mnist' or self.task == 'fashion':
                    for f in row[1:]:
                        tp.append(float(f) / 255)
                    if poison_type == 'backdoor' or poison_type == 'singleshot' or poison_type == 'adaptive':
                        tp[-4:] = [1.0, 1.0, 1.0, 1.0]
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array(tp).reshape(28, 28), axis=0))
                    else:
                        sample_batch.append(np.array(tp))

                elif self.task == 'har':
                    for f in row[1:]:
                        tp.append(float(f) / 2)
                    if poison_type == 'backdoor' or poison_type == 'singleshot' or poison_type == 'adaptive':
                        tp[-3:] = [1.0, 1.0, 1.0, ]
                    if self.model == 'cnn':
                        sample_batch.append(np.expand_dims(np.array([tp]), axis=0))
                    else:
                        sample_batch.append(np.array(tp))
            self.sampleset_testing_poisoned = tc.tensor(sample_batch, device=self.device, dtype=tc.float)
            self.labelset_testing_poisoned = tc.tensor(label_batch, device=self.device, dtype=tc.float)
            sample_batch.clear()
            label_batch.clear()