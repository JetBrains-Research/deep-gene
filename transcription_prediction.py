import random

import lasagne
import theano
import numpy
import theano.tensor as T
from math import exp, log
import time

from conv import get_best_interval
from data import convert_to_number, human_time
from multi_regression_layer import MultiRegressionLayer


def shared_dataset(data_xsy):
    data_x, data_s, data_y = data_xsy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_s = theano.shared(numpy.asarray(data_s, dtype='int8'))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, shared_s, shared_y


def divide_data(interval,
                mask=None):
    left, right = interval

    with open("transcription/tss.fast") as f:
        sequences = [line[left:right] for line in f.readlines()]

    x = []
    y = []

    with open("transcription/abundances.csv") as f:
        for line in f.readlines():
            if (line.startswith("#")):
                continue
            split = line.split("\t")

            if split[2] == "abundance":
                continue

            tpm = float(split[2])

            y.append(log(tpm + exp(-10)))

            x_row = [float(str) for str in split[3:]]
            if mask is not None:
                x_row = [a * b for a, b in zip(x_row, mask)]

            x.append(x_row)

    print("Size of x:{}".format(len(x[0])))

    data = zip(x, sequences, y)

    random.shuffle(data)

    test = data[:5000]
    valid = data[5000:10000]
    train = data[10000:]

    return train, valid, test


def prepare_data(data):
    train, valid, test = data

    def unzip3(l):
        return [[t[i] for t in l] for i in range(3)]

    def prepossess(d):
        binary_data = []
        for (x, s, y) in d:
            binary_data.append((x, convert_to_number(s), y))
        return shared_dataset(unzip3(binary_data))

    return prepossess(train), prepossess(valid), prepossess(test)


class ChipSeqNetwork(object):
    def __init__(self, x, s):
        input = lasagne.layers.InputLayer(shape=(None, 88), input_var=x)
        input_drop = lasagne.layers.DropoutLayer(input, p=0.2)
        layer1 = lasagne.layers.DenseLayer(input_drop, 100, nonlinearity=T.tanh)
        self.output = layer1


def create_conv_input(x, batch_size, sequence_size):
    l_a = T.eq(x, 0).reshape((batch_size, sequence_size))
    l_t = T.eq(x, 1).reshape((batch_size, sequence_size))
    l_g = T.eq(x, 2).reshape((batch_size, sequence_size))
    l_c = T.eq(x, 3).reshape((batch_size, sequence_size))
    return T.cast(T.stack([l_a, l_t, l_g, l_c], axis=1), theano.config.floatX)


class SequenceNetwork(object):
    def __init__(self, x, s, batch_size, sequence_size):
        conv_input = create_conv_input(s, batch_size, sequence_size)

        input = lasagne.layers.InputLayer(shape=(None, 4, sequence_size), input_var=conv_input)

        input_drop = lasagne.layers.DropoutLayer(input, p=0.2)

        conv1 = lasagne.layers.Conv1DLayer(input_drop, num_filters=20, filter_size=4,
                                           nonlinearity=lasagne.nonlinearities.leaky_rectify)

        conv1_pool = lasagne.layers.MaxPool1DLayer(conv1, 2)

        conv1_drop = lasagne.layers.DropoutLayer(conv1_pool, p=0.2)

        conv2 = lasagne.layers.Conv1DLayer(conv1_drop, num_filters=40, filter_size=6,
                                           nonlinearity=lasagne.nonlinearities.leaky_rectify)

        conv2_pool = lasagne.layers.MaxPool1DLayer(conv2, 2)

        conv2_drop = lasagne.layers.DropoutLayer(conv2_pool, p=0.2)

        conv3 = lasagne.layers.Conv1DLayer(conv2_drop, num_filters=60, filter_size=6,
                                           nonlinearity=lasagne.nonlinearities.leaky_rectify)

        conv3_pool = lasagne.layers.MaxPool1DLayer(conv3, 2)

        conv3_drop = lasagne.layers.DropoutLayer(conv3_pool, p=0.2)

        multi_regression_layer = MultiRegressionLayer(conv3_drop,
                                                      nonlinearity=lasagne.nonlinearities.leaky_rectify)

        dence_layer = lasagne.layers.DenseLayer(multi_regression_layer, 100, nonlinearity=None)

        self.output = dence_layer


class Fitter(object):
    def __init__(self,
                 training,
                 validation,
                 test,
                 batch_size,
                 network_type):

        self.batch_size = batch_size
        train_set_x, train_set_s, train_set_y = training
        validation_set_x, validation_set_s, validation_set_y = validation
        test_set_x, test_set_s, test_set_y = test

        x = T.matrix('x')
        s = T.matrix('s', dtype='int8')  # the data is bunch of sequences
        y = T.vector("y")

        self.x = x
        self.s = s
        self.y = y

        index = T.lscalar()  # index to a [mini]batch

        if network_type == "chip-seq":
            chip_network = ChipSeqNetwork(x, s)
            vars_set = {x, y}
            output = chip_network.output
        elif network_type == "sequence":
            seq_network = SequenceNetwork(x, s, batch_size, 1500)
            vars_set = {s, y}
            output = seq_network.output
        elif network_type == "combined":
            chip_network = ChipSeqNetwork(x, s)
            seq_network = SequenceNetwork(x, s, batch_size, 1500)
            output = lasagne.layers.ElemwiseSumLayer([chip_network.output, seq_network.output])
            vars_set = {x, s, y}

        network_output = lasagne.layers.NonlinearityLayer(output,
                                                          nonlinearity=lasagne.nonlinearities.leaky_rectify)

        layer2 = lasagne.layers.DenseLayer(network_output, 100, nonlinearity=lasagne.nonlinearities.leaky_rectify)

        layer2_drop = lasagne.layers.DropoutLayer(layer2, p=0.5)

        regression = lasagne.layers.DenseLayer(layer2_drop, 1, nonlinearity=None)

        output = lasagne.layers.get_output(regression).flatten()

        err = T.mean(lasagne.objectives.squared_error(output, y))

        l1_penalty = lasagne.regularization.regularize_layer_params(regression, lasagne.regularization.l1)
        l2_penalty = lasagne.regularization.regularize_layer_params(regression, lasagne.regularization.l2)

        cost = err + l1_penalty * 1e-4 + l2_penalty * 1e-4

        params = lasagne.layers.get_all_params(regression, trainable=True)

        updates = lasagne.updates.adam(cost, params, learning_rate=0.01)

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens=self.prepare_given(index, vars_set, train_set_x, train_set_s, train_set_y)
        )

        output_deterministic = lasagne.layers.get_output(regression, deterministic=True).flatten()

        err = T.mean(lasagne.objectives.squared_error(output_deterministic, y))

        self.get_validation_error = theano.function(
            inputs=[index],
            outputs=err,
            givens=self.prepare_given(index, vars_set, validation_set_x, validation_set_s, validation_set_y)
        )

        self.get_test_error = theano.function(
            inputs=[index],
            outputs=err,
            givens=self.prepare_given(index, vars_set, test_set_x, test_set_s, test_set_y)
        )

    def prepare_given(self, index, vars_set, set_x, set_s, set_y):
        result = {}

        if self.x in vars_set:
            result[self.x] = set_x[index * self.batch_size:(index + 1) * self.batch_size]

        if self.s in vars_set:
            result[self.s] = set_s[index * self.batch_size:(index + 1) * self.batch_size]

        if self.y in vars_set:
            result[self.y] = set_y[index * self.batch_size:(index + 1) * self.batch_size]

        return result


def get_validation_error(network):
    valid_err = 0.0
    for i in range(5):
        valid_err += network.get_validation_error(i) / 5
    return valid_err


def get_test_error(network):
    test_err = 0.0
    for i in range(5):
        test_err += network.get_test_error(i) / 5
    return test_err


def get_error_from_seq(network_type, data):
    train, validation, test = data
    train_x, train_s, train_y = train
    batch_size = 1000
    train_batches_number = train_x.get_value().shape[0] // batch_size

    network = Fitter(train, validation, test, batch_size, network_type)
    best_error = 1000
    patience = 100
    result_error = 0
    for epoch in range(1000):
        err = 0.0
        for i in range(train_batches_number):
            err += network.train_model(i)

        valid_err = get_validation_error(network)

        print("{:3} total error: {} patience: {}".format(epoch, err / train_batches_number, patience))

        if valid_err < best_error:
            best_error = valid_err
            test_err = get_test_error(network)
            result_error = test_err
            print("      valid_err: {}".format(valid_err))
            print("       test_err: {}".format(test_err))
            patience = 100
        else:
            patience -= 1
            if patience == 0: break

    print result_error
    return result_error


def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    epoch_start = time.time()
    results = {}

    for network_type in ["chip-seq", "sequence", "combined"]:
        data = prepare_data(divide_data(get_best_interval(), mask=None))
        results[network_type] = [get_error_from_seq(network_type, data) for i in range(5)]

    epoch_end = time.time()

    print "time:{}".format(human_time(epoch_end - epoch_start))

    print results


if __name__ == '__main__':
    main()