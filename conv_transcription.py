import random

import theano
import numpy
import theano.tensor as T
from math import exp, log

from adam import adam
from conv import get_best_interval
from data import unzip
from theano_util import HiddenLayer


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = numpy.asarray(data_x, dtype=theano.config.floatX)
    shared_y = numpy.asarray(data_y, dtype=theano.config.floatX)

    return shared_x, shared_y

def divide_data(interval):
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

            x.append([float(str) for str in split[3:]])

    data = zip(x, y)
    test = data[:1000]
    valid = data[1000:2000]
    train = data[2000:]

    return train, valid, test


def prepare_data(data):
    train, valid, test = data

    def prepossess(d):
        binary_data = []
        for (s, t) in d:
            binary_data.append((s, t))
        return shared_dataset(unzip(binary_data))

    return prepossess(train), prepossess(valid), prepossess(test)


class Network(object):
    def __init__(self, rng):
        x = T.matrix('x')
        y = T.vector("y")

        layer0 = HiddenLayer(
            rng,
            x,
            80,
            100)

        layer1 = HiddenLayer(
            rng,
            layer0.output,
            100,
            100)

        layer2 = HiddenLayer(
            rng,
            layer1.output,
            100,
            100,
            activation=None)

        layer3 = HiddenLayer(
            rng,
            layer2.output,
            100,
            1,
            activation=None)

        output = layer3.output.flatten()
        err = T.mean((output - y) ** 2)

        L2 = ((layer0.W ** 2).sum() +
              (layer1.W ** 2).sum() +
              (layer3.W ** 2).sum())

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer3
        self.layer3 = layer3

        params = (layer0.params +
                  layer1.params +
                  layer3.params)

        cost = err + L2 * 0.00001
        updates = adam(cost, params, lr=0.001)

        self.train_model = theano.function(
            inputs=[x, y],
            outputs=cost,
            updates=updates,
        )

        self.get_err = theano.function(
            inputs=[x, y],
            outputs=err,
        )

        self.pedict = theano.function(
            inputs=[x],
            outputs=output,
        )


def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    errors = [get_error() for i in range(5)]
    print errors


def get_error():
    train, valid, test = prepare_data(divide_data(get_best_interval()))
    train_x, train_y = train
    batch_size = 1000
    batches_number = train_x.shape[0] // batch_size
    rng = numpy.random.RandomState(23455)
    network = Network(rng)
    best_error = 1000
    result_error = 0
    for epoch in range(1000):
        err = 0.0
        for i in range(batches_number):
            err += network.train_model(train_x[i * batch_size:(i + 1) * batch_size, ],
                                       train_y[i * batch_size:(i + 1) * batch_size, ])
        valid_err = network.get_err(valid[0], valid[1])
        test_err = network.get_err(test[0], test[1])

        if valid_err < best_error:
            best_error = valid_err
            result_error = test_err
            print("{:3} total error: {}".format(epoch, err / batches_number))
            print("      valid_err: {}".format(valid_err))
            print("       test_err: {}".format(test_err))

    print result_error
    return result_error


if __name__ == '__main__':
    main()