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

from util.logs import get_result_directory_path, FileLogger
from util.logs import human_time


class InputTransformationLayer(lasagne.layers.Layer):
    def __init__(self, incoming, index_var, num_units, num_datasets, name=None, add_nonlinearity=True):
        super(InputTransformationLayer, self).__init__(incoming, name)

        num_inputs = self.input_shape[1]
        self.index_var = index_var
        self.add_nonlinearity = add_nonlinearity
        self.b = self.add_param(lasagne.init.Normal(), (num_datasets, num_inputs, num_units), name="b")
        self.logW = self.add_param(lasagne.init.Normal(mean=-1), (num_datasets, num_inputs, num_units), name="logW")
        self.logC = self.add_param(lasagne.init.Normal(mean=1), (num_datasets, num_inputs, num_units), name="logC")

    def get_output_for(self, input, **kwargs):
        index = self.index_var
        input_dimshuffle = input.dimshuffle([0, 1, 'x'])

        t = T.exp(self.logC[index]) * T.tanh(T.exp(self.logW[index]) * input_dimshuffle + self.b[index])
        if self.add_nonlinearity:
            return T.tanh(T.sum(t, axis=2))
        else:
            return T.sum(t, axis=2)

    def get_output_shape_for(self, input_shape):
        return input_shape


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

genes_number = 57736
dataset_number = len(dataset_names)


def divide_data(index):
    divided_path = os.path.join("data", "transcription", "roadmap", "divided_genes_{}.pkl.gz".format(index))
    if not os.path.exists(divided_path):
        cell = []
        gene = []
        x_final = []
        y_final = []
        for i, name in enumerate(dataset_names):
            x, y = prepare_xy(name)
            assert genes_number == len(y)
            cell.extend([i] * genes_number)
            gene.extend(list(range(genes_number)))
            x_final.extend(x)
            y_final.extend(y)
            print("Size of {} x:{}".format(name, len(x[0])))

        data = zip(cell, gene, x_final, y_final)

        random.shuffle(data)

        test = data[:10000]
        valid = data[10000:20000]
        train = data[20000:]

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


def prepare_data(data):
    train, valid, test = data

    def shared_dataset(data):
        cells, genes, data_x, data_y = data
        shared_c = theano.shared(np.asarray(cells, dtype="int32"))
        shared_g = theano.shared(np.asarray(genes, dtype="int32"))
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

        return shared_c, shared_g, shared_x, shared_y

    def prepossess(d):
        return shared_dataset(zip(*d))

    return prepossess(train), prepossess(valid), prepossess(test)


class Fitter(object):
    def __init__(self,
                 training,
                 validation,
                 test,
                 batch_size,
                 data_set_number):
        self.batch_size = batch_size
        train_set_c, train_set_g, train_set_x, train_set_y = training
        validation_set_c, validation_set_g, validation_set_x, validation_set_y = validation
        test_set_c, test_set_g, test_set_x, test_set_y = test

        c = T.ivector('c')
        g = T.ivector('g')
        x = T.matrix('x')
        y = T.vector('y')

        self.x = x
        self.y = y

        index = T.lscalar()  # index to a [mini]batch

        input_size = 15

        output_layer = self.create_chip_seq_network(batch_size, c, x, data_set_number, input_size)

        l_in_g = lasagne.layers.InputLayer(shape=(batch_size,), input_var=g)
        l_e = lasagne.layers.EmbeddingLayer(l_in_g, genes_number, 1)

        output = lasagne.layers.get_output(output_layer)
        emb = lasagne.layers.get_output(l_e)

        p = T.nnet.sigmoid(emb.reshape((batch_size,)))
        output = output[:, 0] * (1 - p) + output[:, 1] * p

        err = T.mean(lasagne.objectives.squared_error(output, y))

        l1_penalty = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l1)
        l2_penalty = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l2)

        cost = err + l1_penalty * 1e-4 + l2_penalty * 1e-4

        params = lasagne.layers.get_all_params([output_layer, l_e], trainable=True)

        updates = lasagne.updates.adam(cost, params, learning_rate=0.01)

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={c: train_set_c[index * self.batch_size:(index + 1) * self.batch_size],
                    g: train_set_g[index * self.batch_size:(index + 1) * self.batch_size],
                    x: train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: train_set_y[index * self.batch_size:(index + 1) * self.batch_size]}
        )

        output_deterministic = lasagne.layers.get_output(output_layer, deterministic=True)
        output_deterministic = output_deterministic[:, 0] * (1 - p) + output_deterministic[:, 1] * p

        err = T.mean(lasagne.objectives.squared_error(output_deterministic, y), axis=0)

        self.get_validation_error = theano.function(
            inputs=[index],
            outputs=err,
            givens={c: validation_set_c[index * self.batch_size:(index + 1) * self.batch_size],
                    g: validation_set_g[index * self.batch_size:(index + 1) * self.batch_size],
                    x: validation_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: validation_set_y[index * self.batch_size:(index + 1) * self.batch_size]}
        )

        self.get_test_error = theano.function(
            inputs=[index],
            outputs=err,
            givens={c: test_set_c[index * self.batch_size:(index + 1) * self.batch_size],
                    g: test_set_g[index * self.batch_size:(index + 1) * self.batch_size],
                    x: test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: test_set_y[index * self.batch_size:(index + 1) * self.batch_size]}
        )

    def create_chip_seq_network(self, batch_size, c, x, data_set_number, input_size):
        l_in = lasagne.layers.InputLayer(shape=(batch_size, input_size), input_var=x)
        l_t = InputTransformationLayer(l_in, c, 4, data_set_number)

        l_dence1 = lasagne.layers.DenseLayer(l_t, 30, nonlinearity=lasagne.nonlinearities.tanh)

        regression = lasagne.layers.DenseLayer(l_dence1, 2, nonlinearity=None)

        l_t_out = InputTransformationLayer(regression, c, 4, data_set_number, add_nonlinearity=False)

        return l_t_out


def get_validation_error(network):
    valid_err = 0.0
    for i in range(10):
        valid_err += network.get_validation_error(i).mean() / 10
    return valid_err


def get_test_error(network):
    test_err = 0.0
    for i in range(10):
        test_err += network.get_test_error(i) / 10
    return test_err


def get_error_from_seq(data, logger, data_set_number):
    train, validation, test = data
    train_s, train_g, train_x, train_y = train
    batch_size = 1000

    train_batches_number = train_x.get_value().shape[0] // batch_size

    fitter = Fitter(train, validation, test, batch_size, data_set_number)
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
            if patience == 0:
                break

    logger.log(result_error)
    return result_error


def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    result_directory = get_result_directory_path("combined_genes")

    logger = FileLogger(result_directory, "results")

    epoch_start = time.time()

    data_set_num = len(dataset_names)

    for i in range(5):
        data = prepare_data(divide_data(i + 1))
        fitter_logger = FileLogger(result_directory, "log_{}".format(i))

        error = get_error_from_seq(data, fitter_logger, data_set_num)
        fitter_logger.close()
        logger.log("error: {}".format(error))

    epoch_end = time.time()
    logger.log("time:{}".format(human_time(epoch_end - epoch_start)))
    logger.close()


if __name__ == '__main__':
    main()
