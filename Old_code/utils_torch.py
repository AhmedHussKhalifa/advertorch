import json
from torch_model import CNN
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
import numpy as np
import torch
import random

def attack(loss, model, EPS, iteration_k, eps_iter, x_train, y_train):
    if loss == 'xent':
        x_train_adv = projected_gradient_descent(model_fn = model,
                                                x = x_train,
                                                eps = EPS,
                                                eps_iter = eps_iter,
                                                nb_iter = iteration_k,
                                                norm = np.inf,
                                                targeted=False,
                                                # y=y_train,
                                                # clip_min = torch.Tensor(0).to(device="cuda:0"),
                                                # clip_max = torch.Tensor(255).to(device="cuda:0"),
                                            )
    elif loss == 'cw':
        x_train_adv = carlini_wagner_l2(model_fn = model,
                                   x = x_train,
                                   n_classes=10,
                                   max_iterations= 30,
                                   y = y_train,
                                   targeted=False,
                                   lr=EPS/8,
                                   confidence=0.001,
                                   binary_search_steps= 5, 
                                   clip_min = 0 ,
                                   clip_max = 1
                                  )
    

    return x_train_adv


def attack_yang(x_train, y_train, large_num_of_attacks, EPS):
    x_train = x_train.reshape((50, 784))
    x_train = x_train.cpu().detach().numpy()
    y_train = y_train.cpu().detach().numpy()
    x_res =  np.empty((50, 784), int)
    y_res =np.empty((50, ), int)
    for rep in range(large_num_of_attacks):
        shape = x_train.shape
        a=list(np.random.uniform(-EPS,0,1000))
        b= [0] * 1000
        c = [-EPS] * 700
        d = [EPS] * 100
        a.extend(b)
        a.extend(c)
        a.extend(d)
        flattened_array = np.array(random.choices(a,k=shape[0]*shape[1]))
        shaped_array = flattened_array.reshape(shape[0], shape[1])
        x = x_train+shaped_array
        x = np.clip(x, 0, 1)
        x_res = np.concatenate((x_res, x), axis = 0)

        y_res = np.concatenate((y_res, y_train), axis = 0)
    x_train = x_res[50:]
    y_train = y_res[50:]
    x_train = x_train.reshape((5000, 1, 28,28))
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    return x_train, y_train







