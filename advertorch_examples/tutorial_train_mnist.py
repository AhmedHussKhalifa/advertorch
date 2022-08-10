# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
import numpy as np


class CNN_Model(nn.Module):
    # Constructor
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, 
                              kernel_size=5, padding = 2)
        self.maxpool1=nn.MaxPool2d(kernel_size = 2, padding = 1, stride = 2)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=5, padding = 2)
        self.maxpool2=nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(64* 7* 7, 1024)
        self.fc1_bn=nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data,mean=0, std=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.1)
            nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    # # first fully connected layer 
    # W_fc1 = self._weight_variable([7 * 7 * 64, 1024]) 
    # b_fc1 = self._bias_variable([1024]) 
 
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) 
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 
    # #batch norm 
    # batch_mean2, batch_var2 = tf.nn.moments(h_fc1, [0]) 
    # scale2 = tf.Variable(tf.ones([1024])) 
    # beta2 = tf.Variable(tf.zeros([1024])) 
    # batch_norm = tf.nn.batch_normalization(h_fc1,batch_mean2,batch_var2,beta2,scale2,0.3) 
    # batch_norm_sig  = tf.nn.sigmoid(batch_norm) 


    # Prediction
    def forward(self, x):
        # print("original Image Shape: ", x.shape)
        x = self.cnn1(x)
        # print("CNN 1 Shape: ", x.shape)
        x = torch.relu(x)
        # print("Relu 1 Shape: ", x.shape)
        x = self.maxpool1(x)
        # print("MaxPool 1 Shape: ", x.shape)
        x = self.cnn2(x)
        # print("CNN 2 Shape: ", x.shape)
        x = torch.relu(x)
        # print("Relu 2 Shape: ", x.shape)
        x = self.maxpool2(x)
        # print("MaxPool 2 Shape: ", x.shape)
        # Flatten the matrices
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print("FC 1 Shape: ", x.shape)

        x = torch.relu(x)
        x = self.fc1_bn(x)

        x = self.fc2(x)
        # print("FC 2 Shape: ", x.shape)
        return x

def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def log_softmax(x):
    return x - torch.logsumexp(x,dim=1, keepdim=True)

def my_CrossEntropyLoss(outputs, targets, lamda=1):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]
    outputs = - torch.sum(outputs)/num_examples
    outputs = torch.exp(lamda * outputs)
    return outputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="cln", help="cln | adv ")
    parser.add_argument('--expo_mode', default="adv", help="yang")
    parser.add_argument('--train_batch_size', default=50, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 10
        model_filename = "mnist_CNN_Model_clntrained.pt"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 84
        model_filename = "mnist_CNN_Model_advtrained.pt"
    else:
        raise

    train_loader = get_mnist_train_loader(
        batch_size=args.train_batch_size, shuffle=True)
    test_loader = get_mnist_test_loader(
        batch_size=args.test_batch_size, shuffle=False)

    # model = LeNet5()

    model = CNN_Model()
    model.initialize_weights(model)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    large_num_of_attacks = 1
    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack, YangAttack
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
        if args.expo_mode in "yang":
            large_num_of_attacks = 100
            print("Yang Attack:  large_num_of_attacks =  ", large_num_of_attacks)
            adversary_yang = YangAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=False, large_num_of_attacks=large_num_of_attacks)

    itr = 0
    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ori = data
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should NOT be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    if args.expo_mode == "yang":
                        data, target = adversary_yang.perturb(data, target)
                        data, target = data.to(device), target.to(device)
                    else:
                        data = adversary.perturb(data, target)
                    
            optimizer.zero_grad()
            output = model(data)
            
            if args.expo_mode == "yang":
                loss = my_CrossEntropyLoss(output, target, lamda=1)
            else:
                loss = F.cross_entropy(output, target, reduction='elementwise_mean')
            
            loss.backward()
            optimizer.step()
            # if batch_idx % args.log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx *
            #         len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
            if itr % 300 == 0:
                print('Train Epoch: {} || Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, itr, batch_idx *
                    len(data)/large_num_of_attacks , len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            itr += 1

        model.eval()
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename))
