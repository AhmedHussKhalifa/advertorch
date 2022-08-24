"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import shutil
import math
from timeit import default_timer as timer
import pickle
import tensorflow as tf
import sys
import time
# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model_train import Model
from pgd_attack import LinfPGDAttack, YangAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']
#eval
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
global_step_eval = tf.contrib.framework.get_or_create_global_step()


# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

attack_yang = YangAttack(config['epsilon'])
# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
#eval changes
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)
last_checkpoint_filename = ''
already_seen_state = False

saver_eval = tf.train.Saver()
summary_writer_eval = tf.summary.FileWriter(eval_dir)
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver_eval.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):

        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = mnist.test.images[bstart:bend, :]
        y_batch = mnist.test.labels[bstart:bend]

        dict_nat = {model.x_input: x_batch,
              model.y_input: y_batch}

        x_batch_adv = attack.perturb(x_batch, y_batch, sess)

        dict_adv = {model.x_input: x_batch_adv,
              model.y_input: y_batch}

        cur_corr_nat, cur_xent_nat = sess.run(
                                  [model.num_correct,model.xent],
                                  feed_dict = dict_nat)
        cur_corr_adv, cur_xent_adv = sess.run(
                                  [model.num_correct,model.xent],
                                  feed_dict = dict_adv)

        total_xent_nat += cur_xent_nat
        total_xent_adv += cur_xent_adv
        total_corr_nat += cur_corr_nat
        total_corr_adv += cur_corr_adv
        # print("total_corr_adv : ", ibatch)
    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / (num_eval_examples)
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / (num_eval_examples)

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer_eval.add_summary(summary, global_step_eval.eval(sess))

    print('Test natural: {:.2f}%'.format(100 * acc_nat))
    print('Test adversarial: {:.2f}%'.format(100 * acc_adv))
    print('Test avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('Test avg adv loss: {:.4f}'.format(avg_xent_adv))

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)
large_num_of_attacks = config['large_num_of_attacks']


# if tf.test.gpu_device_name():
#   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# exit(0)
with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0


  # for ii in range(max_num_training_steps):
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size) # 50 x 784

    

    # Compute Adversarial Perturbations

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}
    start = timer()
    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    

  
    x_batch_adv, y_batch = attack_yang.perturb(x_batch, y_batch, large_num_of_attacks)
    end = timer()
    training_time += end - start
    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
        nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
        print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
        if ii != 0:
            print('    {} examples per second'.format(
                num_output_steps * batch_size / training_time))
            training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
        summary = sess.run(merged_summaries, feed_dict=adv_dict)
        summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
        saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)
        print("saving new checkpoint")
    # Actual training step
    sess.run(train_step, feed_dict=adv_dict)


    # y_xent = sess.run(model.y_xent, feed_dict=adv_dict)
    # y_xent_list = list(y_xent)
    # y_xent_max = [max(y_xent_list[i:i + 201]) for i in range(0, 5000, 200)]
    #y_xent_sum = [sum(y_xent_list[i:i + 201]) for i in range(0, 5000, 200)]
    # softmax = sess.run(model.softmax, feed_dict=adv_dict)

    # print("#"*50)
    # # print(type(y_xent))
    # # tf.math.reduce_max(
    # print("max :",  max(y_xent_max))
    # print("sum :",  max(y_xent_sum))


    # print("After", np.min(softmax) , np.max(softmax), np.mean(softmax))



    # avg_softmax.append(np.mean(softmax))
    # min_softmax.append(np.min(softmax))
    # min_softmax.append(np.max(softmax))


    # print("#"*50)


    # correct_prediction = sess.run(model.correct_prediction, feed_dict=adv_dict)
    # y_xent = sess.run(model.y_xent, feed_dict=adv_dict)
    # print(y_pred)
    # print(y_batch)
    # print(correct_prediction)

    cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    # print("cur_checkpoint: ", cur_checkpoint)
    # Case 1: No checkpoint yet
    if cur_checkpoint is None:
        if not already_seen_state:
            print('No checkpoint yet, waiting ...', end='')
            already_seen_state = True
        else:

            print('first else in first if.', end='')
        sys.stdout.flush()
    # Case 2: Previously unseen checkpoint
    elif cur_checkpoint != last_checkpoint_filename:
        print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                              datetime.now()))
        sys.stdout.flush()
        last_checkpoint_filename = cur_checkpoint
        already_seen_state = False
        evaluate_checkpoint(cur_checkpoint)
    # Case 3: Previously evaluated checkpoint
    else:
        if not already_seen_state:
            print('Waiting for the next checkpoint ...   ({})   '.format(
                datetime.now()),
                end='')
            already_seen_state = True
        # else:
        #     print('.', end='')
        sys.stdout.flush()


