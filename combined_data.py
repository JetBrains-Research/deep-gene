import gzip
import random
import time
from math import exp, log

import cPickle
import lasagne
import numpy as np
import os
import theano
import theano.tensor as T

from independet_prediction import InputTransformationLayer
from util.logs import get_result_directory_path, FileLogger
from util.logs import human_time


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, shared_y


dataset_names = ["esophagus",
              "gastric",
              "H1_cell_line",
              "H1_derived_mesenchymal_stem_cells",
              "H1_derived_neuronal_progenitor_cultured_cells",
              "IMR90_cell_line",
              "iPS_DF_6.9_cell_line",
              "iPS_DF_19.11_cell_line",
              "pancreas",
              "sigmoid_colon",
              "small_intestine",
              "spleen"]


def divide_data(index):
    divided_path = os.path.join("data", "transcription", "roadmap", "divided_{}.pkl.gz".format(index))
    if not os.path.exists(divided_path):
        x_list = []
        y_list = []
        for name in dataset_names:
            x, y = prepare_xy(name)
            x_list.append(x)
            y_list.append(y)
            print("Size of {} x:{}".format(name, len(x[0])))

        x_final = zip(*x_list)
        y_final = zip(*y_list)

        data = zip(x_final, y_final)

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


def prepare_xy(name):
    x = []
    y = []
    with open(os.path.join("data", "transcription", "roadmap", name + ".csv")) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            split = line.split("\t")

            if split[0] == "gene":
                continue

            tpm = float(split[1])

            y.append(log(tpm + exp(-8)))
            x_row = [float(s) for s in split[2:]]
            x.append(x_row)
    return x, y


def prepare_data(data, sclice=None):
    train, valid, test = data

    def prepossess(d):
        x, y = zip(*d)
        if sclice is not None:
            x = [t[sclice:sclice + 1] for t in x]
            y = [t[sclice:sclice + 1] for t in y]
            return shared_dataset((x, y))
        else:
            return shared_dataset((x, y))

    return prepossess(train), prepossess(valid), prepossess(test)


class Fitter(object):
    def __init__(self,
                 training,
                 validation,
                 test,
                 batch_size,
                 data_set_size):
        self.batch_size = batch_size
        train_set_x, train_set_y = training
        validation_set_x, validation_set_y = validation
        test_set_x, test_set_y = test

        x = T.tensor3('x')
        y = T.matrix('y')

        self.x = x
        self.y = y

        index = T.lscalar()  # index to a [mini]batch

        input_size = 15

        output_layer = self.create_chip_seq_network(batch_size, x, data_set_size, input_size)

        output = lasagne.layers.get_output(output_layer)

        err = T.mean(lasagne.objectives.squared_error(output, y))

        l1_penalty = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l1)
        l2_penalty = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l2)

        cost = err + l1_penalty * 1e-4 + l2_penalty * 1e-4

        params = lasagne.layers.get_all_params(output_layer, trainable=True)

        updates = lasagne.updates.adam(cost, params, learning_rate=0.01)

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={x: train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: train_set_y[index * self.batch_size:(index + 1) * self.batch_size]}
        )

        output_deterministic = lasagne.layers.get_output(output_layer, deterministic=True)

        err = T.mean(lasagne.objectives.squared_error(output_deterministic, y), axis=0)

        self.get_validation_error = theano.function(
            inputs=[index],
            outputs=err,
            givens={x: validation_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: validation_set_y[index * self.batch_size:(index + 1) * self.batch_size]}
        )

        self.get_test_error = theano.function(
            inputs=[index],
            outputs=err,
            givens={x: test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: test_set_y[index * self.batch_size:(index + 1) * self.batch_size]}
        )

    def create_chip_seq_network(self, batch_size, x, data_set_size, input_size):
        l_in = lasagne.layers.InputLayer(shape=(batch_size, data_set_size, input_size), input_var=x)

        l_reshape = lasagne.layers.ReshapeLayer(l_in, (batch_size, data_set_size * input_size))

        l_t = InputTransformationLayer(l_reshape, 4)

        l_reshape2 = lasagne.layers.ReshapeLayer(l_t, (batch_size * data_set_size, input_size))

        l_dence1 = lasagne.layers.DenseLayer(l_reshape2, 30, nonlinearity=lasagne.nonlinearities.tanh)

        regression = lasagne.layers.DenseLayer(l_dence1, 1, nonlinearity=None)

        l_reshape3 = lasagne.layers.ReshapeLayer(regression, (batch_size, data_set_size))

        l_t_out = InputTransformationLayer(l_reshape3, 4, add_nonlinearity=False)

        return l_t_out


def get_validation_error(network):
    valid_err = 0.0
    for i in range(5):
        valid_err += network.get_validation_error(i).mean() / 5
    return valid_err


def get_test_error(network):
    test_err = 0.0
    for i in range(5):
        test_err += network.get_test_error(i) / 5
    return test_err


def get_error_from_seq(data, logger, data_set_size):
    train, validation, test = data
    train_x, train_y = train
    batch_size = 1000

    train_batches_number = train_x.get_value().shape[0] // batch_size

    fitter = Fitter(train, validation, test, batch_size, data_set_size)
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
            patience = 100

        else:
            patience -= 1
            if patience == 0: break

    logger.log(result_error)
    return result_error


def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    result_directory = get_result_directory_path("combined_prediction")

    logger = FileLogger(result_directory, "results")

    epoch_start = time.time()

    data_set_num = len(dataset_names)

    errors = np.zeros((data_set_num, 5))

    for s in range(data_set_num):
        for i in range(5):
            data = prepare_data(divide_data(i + 1), s)
            logger.log("start: {} {}".format(dataset_names[s], i))
            fitter_logger = FileLogger(result_directory, "log_{}_{}".format(dataset_names[s], i))

            error = get_error_from_seq(data, fitter_logger, 1)
            fitter_logger.close()
            logger.log("error: {}".format(error))
            errors[s, i] = error

    for s in range(data_set_num):
        logger.log(dataset_names[s])
        logger.log(errors[s])

    errors = np.zeros((data_set_num, 5))

    for i in range(5):
        data = prepare_data(divide_data(i + 1))
        fitter_logger = FileLogger(result_directory, "log_{}".format(i))

        error = get_error_from_seq(data, fitter_logger, data_set_num)
        errors[:, i] = error
        fitter_logger.close()
        logger.log("error: {}".format(error))
        logger.log("mean: {}".format(error.mean()))

    for s in range(data_set_num):
        logger.log(dataset_names[s])
        logger.log(errors[s])

    epoch_end = time.time()
    logger.log("time:{}".format(human_time(epoch_end - epoch_start)))
    logger.close()


if __name__ == '__main__':
    main()
