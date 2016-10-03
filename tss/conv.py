import os
import sys
import gzip
import cPickle

import numpy
import theano
import theano.tensor as T
import time

from util.adam import adam
from conv_model import Network
from util.data import shared_dataset, unzip, human_time, convert_to_number


def prepare_data(data, left, right):
    train, valid, test = data

    def prepossess(d):
        binary_data = []
        for (s, t) in d:
            binary_data.append((convert_to_number(s[left:right]), t))
        return shared_dataset(unzip(binary_data))

    return prepossess(train), prepossess(valid), prepossess(test)


class Fitter():
    def __init__(self,
                 network,
                 training,
                 validation,
                 test,
                 batch_size,
                 learning_rate,
                 L1_reg_coef,
                 L2_reg_coef):
        train_set_x, train_set_y = training
        validation_set_x, validation_set_y = validation
        test_set_x, test_set_y = test

        self.network = network
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        self.n_validation_batches = validation_set_x.get_value(borrow=True).shape[0] // batch_size

        x = network.x
        y = T.bvector('y')  # the labels are presented as 1D vector of

        index = network.index
        is_train = network.is_train

        L1 = (abs(network.conv_1.W).sum() +
              abs(network.conv_2.W).sum() +
              abs(network.conv_3.W).sum() +
              abs(network.mr_layer.W).sum() +
              abs(network.fully_connected.W).sum() +
              abs(network.regression.W).sum())

        L2 = ((network.conv_1.W ** 2).sum() +
              (network.conv_2.W ** 2).sum() +
              (network.conv_3.W ** 2).sum() +
              (network.mr_layer.W ** 2).sum() +
              (network.fully_connected.W ** 2).sum() +
              (network.regression.W ** 2).sum())

        # the cost we minimize during training is the NLL of the model
        cost = network.regression.negative_log_likelihood(y) + L1_reg_coef * L1 + L2_reg_coef * L2

        # create a list of all model parameters to be fit by gradient descent
        params = (network.conv_1.params +
                  network.conv_2.params +
                  network.conv_3.params +
                  network.mr_layer.params +
                  network.fully_connected.params +
                  network.regression.params)

        updates = adam(cost, params, lr=learning_rate)

        self.train_model = theano.function(
            inputs=[index],
            outputs=(cost, network.regression.errors(y)),
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: numpy.cast['int32'](1)
            }
        )

        self.test_error = theano.function(
            inputs=[index],
            outputs=network.regression.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: numpy.cast['int32'](0)
            }
        )

        self.validation_error = theano.function(
            inputs=[index],
            outputs=network.regression.errors(y),
            givens={
                x: validation_set_x[index * batch_size:(index + 1) * batch_size],
                y: validation_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: numpy.cast['int32'](0)
            }
        )

    def get_test_error(self):
        n = self.n_test_batches
        return sum([self.test_error(i) for i in range(n)]) / float(n)

    def get_validation_error(self):
        n = self.n_validation_batches
        return sum([self.validation_error(i) for i in range(n)]) / float(n)

    def do_fit(self, log_path, model_path=None):
        test_start = time.time()
        patience = 100
        step = 1
        best_error = 1.0
        test_error = 0

        n_train_batches = self.n_train_batches

        log = open(log_path, 'w')
        for epoch in range(1000):
            a_cost = 0.0
            a_train_error = 0.0
            best_valid_error = 1.0
            epoch_start = time.time()
            for minibatch_index in xrange(n_train_batches):
                sys.stdout.write("*")
                sys.stdout.flush()
                (cost, train_error) = self.train_model(minibatch_index)
                a_cost += cost / n_train_batches
                a_train_error += train_error / n_train_batches
                if step % (n_train_batches // 5) == 0:
                    validation_error = self.get_validation_error()

                    if validation_error < best_valid_error:
                        best_valid_error = validation_error

                    if validation_error < best_error:
                        if model_path is not None:
                            with gzip.open(model_path, 'w') as f:
                                cPickle.dump(self.network.save_state(), f)
                        patience = 100
                        best_error = validation_error
                        test_error = self.get_test_error()
                    else:
                        patience -= 1
                step += 1

            print ""
            if patience < 0:
                break

            epoch_end = time.time()
            line = (("epoch: {}, cost: {:.5f}, train err: {:.5f}, valid err: {:.5f}, " +
                     "best err: {:.5f}, test err: {:.5f}, time: {}")
                    .format(epoch, a_cost, a_train_error, best_valid_error,
                            best_error, test_error, human_time(epoch_end - epoch_start)))
            print(line)
            log.write(line + "\n")

            epoch += 1
        test_end = time.time()
        log.write("result: {}\n".format(test_error))
        log.write("time: {}\n".format(human_time(test_end - test_start)))
        log.close()
        return test_error


def create_default_network(batch_size=1000, parameters=None):
    rng = numpy.random.RandomState(23455)
    if not parameters:
        parameters = get_default_parameters()
    model = Network(rng, batch_size, parameters)
    return model


def get_model_name(data_name, index):
    return "best_conv_model_{}_{}".format(data_name, index)


def get_model_parameters_path(dataset_name, index):
    model_name = get_model_name(dataset_name, index)
    model_path = 'models/{}.pkl.gz'.format(model_name)
    return model_path


def get_best_interval():
    return 1000, 2500


def get_default_parameters():
    return {
        "left": 1000,
        "right": 2500,
        "n_kernels1": 40,
        "n_kernels2": 60,
        "n_kernels3": 100,
        "n_fully_connected": 60,
        "pattern1_size": 4,
        "pattern2_size": 6,
        "pattern3_size": 6,
        "dropout0": 0.1,
        "dropout1": 0.2,
        "dropout2": 0.2,
        "dropout3": 0.2,
        "dropout4": 0.5,
        "dropout5": 0.1,
        "learning_rate": 0.001,
        "L1_reg_coef": 0.00001,
        "L2_reg_coef": 0.00001,
    }


def get_dataset_types_mm9():
    return ["mm9_genes_coding", "mm9_genes_all", "mm9_cage_near_coding", "mm9_cage_all"]


def get_dataset_types_hg19():
    return ["hg19_genes_coding", "hg19_genes_all", "hg19_cage_near_coding", "hg19_cage_all"]


def get_dataset_types():
    return get_dataset_types_mm9() + get_dataset_types_hg19()

