import gzip
import random
import time
from math import exp, log

import cPickle
import lasagne
import numpy as np
import pandas as pd
import os
import theano
import theano.tensor as T

from util.data import convert_to_number
from util.logs import get_result_directory_path, PrintLogger, FileLogger
from util.logs import human_time
from util.multi_regression_layer import MultiRegressionLayer


def shared_dataset(data_xsy):
    data_x, data_s, data_y = data_xsy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_s = theano.shared(np.asarray(data_s, dtype='int8'))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, shared_s, shared_y

input_size = 89

def divide_data(name, index):
    divided_path = os.path.join("data", "transcription", name, "divided_{}.pkl.gz".format(index))
    if not os.path.exists(divided_path):
        with open(os.path.join("data", "transcription", name, "tss.fast")) as f:
            sequences = [line[:-1] for line in f.readlines()]

        x = []
        y = []

        with open(os.path.join("data", "transcription", name, "abundances.csv")) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                split = line.split("\t")

                if split[2] == "abundance":
                    continue

                tpm = float(split[2])

                y.append(log(tpm + exp(-8)))

                x_row = [float(str) for str in split[3:]]

                x.append(x_row)

        print("Size of x:{}".format(len(x[0])))
        print("y interval {} {}".format(min(y), max(y)))

        data = zip(x, sequences, y)

        random.shuffle(data)

        test = data[:5000]
        valid = data[5000:10000]
        train = data[10000:]

        result = (train, valid, test)

        print("Start writing: {}".format(divided_path))
        with gzip.open(divided_path, 'w') as f:
            cPickle.dump(result, f)
        print("Done")

    print("Start reading: {}".format(divided_path))
    with gzip.open(divided_path, 'r') as f:
        data = cPickle.load(f)
        print("Done")
        return data


def prepare_data(data, left, right, mask=None):
    train, valid, test = data

    def unzip3(l):
        return [[t[i] for t in l] for i in range(3)]

    def apply_mask(x_row):
        if mask:
            return [a * b for a, b in zip(x_row, mask)]
        else:
            return x_row

    def prepossess(d):
        binary_data = []
        for (x, s, y) in d:
            binary_data.append((apply_mask(x), convert_to_number(s[left:right]), y))
        return shared_dataset(unzip3(binary_data))

    return prepossess(train), prepossess(valid), prepossess(test)


class ChipSeqNetwork(object):
    def __init__(self, x, input_size):
        input = lasagne.layers.InputLayer(shape=(None, input_size), input_var=x)
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
    def __init__(self, s, batch_size, sequence_size):
        conv_input = create_conv_input(s, batch_size, sequence_size)

        input = lasagne.layers.InputLayer(shape=(None, 4, sequence_size), input_var=conv_input)

        input_drop = lasagne.layers.DropoutLayer(input, p=0.2)

        conv1 = lasagne.layers.Conv1DLayer(input_drop, num_filters=20, filter_size=4,
                                           nonlinearity=lasagne.nonlinearities.tanh)

        conv1_pool = lasagne.layers.MaxPool1DLayer(conv1, 2)

        conv1_drop = lasagne.layers.DropoutLayer(conv1_pool, p=0.2)

        conv2 = lasagne.layers.Conv1DLayer(conv1_drop, num_filters=40, filter_size=6,
                                           nonlinearity=lasagne.nonlinearities.tanh)

        conv2_pool = lasagne.layers.MaxPool1DLayer(conv2, 2)

        conv2_drop = lasagne.layers.DropoutLayer(conv2_pool, p=0.2)

        conv3 = lasagne.layers.Conv1DLayer(conv2_drop, num_filters=60, filter_size=6,
                                           nonlinearity=lasagne.nonlinearities.tanh)

        conv3_pool = lasagne.layers.MaxPool1DLayer(conv3, 2)

        conv3_drop = lasagne.layers.DropoutLayer(conv3_pool, p=0.2)

        multi_regression_layer = MultiRegressionLayer(conv3_drop,
                                                      nonlinearity=lasagne.nonlinearities.tanh)

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
            output_layer, vars_set = self.create_chip_seq_network(s, x, y)

        elif network_type == "sequence":
            output_layer, vars_set = self.creaty_sequence_network(s, x, y)

        elif network_type == "combined":
            output_layer, vars_set = self.create_combined_network(s, x, y)

        else:
            print("Unexpected network type:{}".format(network_type))
            raise

        network_output = lasagne.layers.NonlinearityLayer(output_layer,
                                                          nonlinearity=lasagne.nonlinearities.tanh)

        layer2 = lasagne.layers.DenseLayer(network_output,
                                           100,
                                           nonlinearity=lasagne.nonlinearities.tanh)

        layer2_drop = lasagne.layers.DropoutLayer(layer2, p=0.5)

        regression = lasagne.layers.DenseLayer(layer2_drop, 1, nonlinearity=None)

        output = lasagne.layers.get_output(regression).flatten()

        if network_type == "chip-seq":
            get_output_internal = theano.function(
                inputs=[x],
                outputs=output
            )

            def get_output(x_v, s_v):
                return get_output_internal(x_v)

            self.get_output = get_output
        elif network_type == "sequence":
            get_output_internal = theano.function(
                inputs=[s],
                outputs=output,
            )

            def get_output(x_v, s_v):
                return get_output_internal(s_v)

            self.get_output = get_output

        elif network_type == "combined":
            get_output_internal = theano.function(
                inputs=[x, s],
                outputs=output,
            )

            def get_output(x_v, s_v):
                return get_output_internal(x_v, s_v)

            self.get_output = get_output

        else:
            print("Unexpected network type:{}".format(network_type))
            raise

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

    def create_chip_seq_network(self, s, x, y):
        chip_network = ChipSeqNetwork(x, input_size)
        vars_set = {x, y}
        output = chip_network.output
        return output, vars_set

    def creaty_sequence_network(self, s, x, y):
        seq_network = SequenceNetwork(s, self.batch_size, 1500)
        vars_set = {s, y}
        output = seq_network.output
        return output, vars_set

    def create_combined_network(self, s, x, y):
        chip_network = ChipSeqNetwork(x, input_size)
        seq_network = SequenceNetwork(s, self.batch_size, 1500)
        output = lasagne.layers.ElemwiseSumLayer([chip_network.output, seq_network.output])
        vars_set = {x, s, y}
        return output, vars_set

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


def write_prediction(fitter, train_x, train_s, train_y, path):
    batch_size = fitter.batch_size
    train_x_value = train_x.get_value()
    train_s_value = train_s.get_value()
    train_y_value = train_y.get_value()
    train_batches_number = train_x_value.shape[0] // batch_size

    tpm = np.zeros((train_batches_number * batch_size))
    pred_tpm = np.zeros((train_batches_number * batch_size))

    for i in range(train_batches_number):
        offset = i * batch_size
        tpm[offset:offset + batch_size] = train_y_value[offset:offset + batch_size]
        pred_tpm[offset:offset + batch_size] = fitter.get_output(
            train_x_value[offset:offset + batch_size],
            train_s_value[offset:offset + batch_size])

    df = pd.DataFrame({'tpm': tpm,
                       'pred_tpm': pred_tpm})

    df.to_csv(path)


def get_error_from_seq(network_type, data, logger, result_path=None):
    train, validation, test = data
    train_x, train_s, train_y = train
    batch_size = 1000
    train_batches_number = train_x.get_value().shape[0] // batch_size

    fitter = Fitter(train, validation, test, batch_size, network_type)
    best_error = 1000
    patience = 100
    result_error = 0
    for epoch in range(1000):
        err = 0.0
        for i in range(train_batches_number):
            err += fitter.train_model(i)

        valid_err = get_validation_error(fitter)

        logger.log("{:3} total error: {:.3f} valid error {:.3f} patience: {}".format(
            epoch,
            err / train_batches_number,
            valid_err,
            patience))

        if valid_err < best_error:
            best_error = valid_err
            test_err = get_test_error(fitter)
            result_error = test_err
            logger.log("      valid_err: {}".format(valid_err))
            logger.log("       test_err: {}".format(test_err))
            if result_path:
                write_prediction(fitter, train_x, train_s, train_y, result_path)
            patience = 100

        else:
            patience -= 1
            if patience == 0: break

    logger.log(result_error)
    return result_error


def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    result_directory = get_result_directory_path("transcription_prediction")

    logger = FileLogger(result_directory, "results")

    for network_type in ["chip-seq", "sequence", "combined"]:
        epoch_start = time.time()
        logger.log("start {}".format(network_type))

        for i in range(5):
            data = prepare_data(divide_data("CD4", i + 1), 1000, 2500)
            tpm_path = os.path.join(result_directory, "tpm_pred_{}_{}.csv".format(network_type, i))
            fitter_logger = FileLogger(result_directory, "{}_{}".format(network_type, i))
            error = get_error_from_seq(network_type, data, fitter_logger, tpm_path)
            fitter_logger.close()
            logger.log("error: {}".format(error))

        epoch_end = time.time()
        logger.log("time:{}".format(human_time(epoch_end - epoch_start)))

    logger.close()


if __name__ == '__main__':
    main()
