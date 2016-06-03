import lasagne
import theano

import theano.tensor as T

import numpy
import time

from lasagne.layers import get_output

from adam import adam
from data import human_time
from logistic import LogisticRegression
from theano_util import HiddenLayer


class TheanoModel(object):
    def __init__(self,
                 rng,
                 batch_size,
                 train_set_x,
                 train_set_y):
        x = T.matrix("x")
        y = T.ivector("y")
        is_train = T.iscalar('is_train')
        index = T.lscalar()  # index to a [mini]batch

        self.x = x
        self.y = y

        layer1 = HiddenLayer(
            rng=rng,
            input=x,
            n_in=1000,
            n_out=1000,
            activation=T.tanh)



        layer2 = HiddenLayer(
            rng=rng,
            input=layer1.output,
            n_in=1000,
            n_out=1000,
            activation=T.tanh)

        regression = LogisticRegression(
            input=layer2.output,
            n_in=1000,
            n_out=2)

        L1 = (abs(layer1.W).sum() +
              abs(layer2.W).sum() +
              abs(regression.W).sum())

        L2 = ((layer1.W ** 2).sum() +
              (layer2.W ** 2).sum() +
              (regression.W ** 2).sum())

        cost = regression.negative_log_likelihood(y) + 1e-4 * L1 + 1e-4 * L2

        # create a list of all model parameters to be fit by gradient descent
        params = (layer1.params +
                  layer2.params +
                  regression.params)

        updates = adam(cost, params, lr=0.001)

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                #is_train: numpy.cast['int32'](1)
            }
        )

class LasagneModel(object):
    def __init__(self,
                 rng,
                 batch_size,
                 train_set_x,
                 train_set_y):
        x = T.matrix("x")
        y = T.ivector("y")
        is_train = T.iscalar('is_train')
        index = T.lscalar()  # index to a [mini]batch

        self.x = x
        self.y = y

        input = lasagne.layers.InputLayer(shape=(None, 1000), input_var=x)

        input_drop = lasagne.layers.DropoutLayer(input, p=0.2)

        layer1 = lasagne.layers.DenseLayer(input_drop, 1000, nonlinearity=T.tanh)

        layer1_drop = lasagne.layers.DropoutLayer(layer1, p=0.5)

        layer2 = lasagne.layers.DenseLayer(layer1_drop, 1000, nonlinearity=T.tanh)

        layer2_drop = lasagne.layers.DropoutLayer(layer2, p=0.5)

        regression = lasagne.layers.DenseLayer(
            layer2_drop,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

        predictions = get_output(regression)
        loss = lasagne.objectives.categorical_crossentropy(predictions, y).mean()

        l1_penalty = lasagne.regularization.regularize_layer_params(regression, lasagne.regularization.l1)
        l2_penalty = lasagne.regularization.regularize_layer_params(regression, lasagne.regularization.l2)

        cost = loss + l1_penalty * 1e-4 + l2_penalty * 1e-4

        params = lasagne.layers.get_all_params(regression, trainable=True)

        updates = lasagne.updates.adam(cost, params, learning_rate=0.001)

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
            }
        )



def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    x = numpy.random.rand(100000, 1000)
    y = numpy.random.randint(2, size=100000)

    shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(y, dtype='int32'))

    rng = numpy.random.RandomState(23455)

    model = TheanoModel(rng, 1000, shared_x, shared_y)
    fit(model)

    model = LasagneModel(rng, 1000, shared_x, shared_y)
    fit(model)


def fit(model):
    for epoch in range(10):
        epoch_start = time.time()

        err = 0

        for batch in range(100):
            err += model.train_model(batch) / 100

        print "err:{}".format(err)

        epoch_end = time.time()

        print "time:{}".format(human_time(epoch_end - epoch_start))


if __name__ == '__main__':
    main()
