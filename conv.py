import random
import sys
import gzip
import cPickle

import numpy
import theano
import theano.tensor as T
import time

from adam import adam
from logistic import LogisticRegression
from theano_util import LeNetConvPoolLayer, HiddenLayer, add_dropout, relu
from data import divide_data, convert_to_binary_layered, shared_dataset, unzip, human_time


def prepare_data(data, interval):
    left, right = interval
    train, valid, test = data

    def prepossess(d):
        binary_data = []
        for (s, t) in d:
            binary_data.append((convert_to_binary_layered(s[left:right]), t))
        return shared_dataset(unzip(binary_data))

    return prepossess(train), prepossess(valid), prepossess(test)


def get_pattern_function1(batch_size, sequence_size, pattern1_size, n_kernels1, conv_1_params):
    rng = numpy.random.RandomState(23455)

    x = T.tensor4('x')  # the data is bunch of 3D patterns

    conv_1 = LeNetConvPoolLayer(
        rng,
        input=x,
        image_shape=(batch_size, 4, sequence_size, 1),
        filter_shape=(n_kernels1, 4, pattern1_size, 1),
        poolsize=(1, 1)
    )

    conv_1.load_state(conv_1_params)

    return theano.function(
        [x],
        conv_1.output
    )


def get_pattern_function2(batch_size, sequence_size, pattern1_size, pattern2_size, n_kernels1, n_kernels2,
                          layer0_params, layer1_params):
    rng = numpy.random.RandomState(23455)

    x = T.tensor4('x')  # the data is bunch of 3D patterns

    layer0 = LeNetConvPoolLayer(
        rng,
        input=x,
        image_shape=(batch_size, 4, sequence_size, 1),
        filter_shape=(n_kernels1, 4, pattern1_size, 1),
        poolsize=(2, 1)
    )

    layer0.load_state(layer0_params)

    layer0_out_size = (sequence_size - pattern1_size + 1) / 2

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, n_kernels1, layer0_out_size, 1),
        filter_shape=(n_kernels2, n_kernels1, pattern2_size, 1),
        poolsize=(1, 1)
    )

    layer1.load_state(layer1_params)

    return theano.function(
        [x],
        layer1.output
    )


class MultiRegressionLayer(object):
    def __init__(self, rng, input, n_out, n_seq):
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_out + n_seq)),
                high=numpy.sqrt(6. / (n_out + n_seq)),
                size=(n_out, n_seq)
            ),
            dtype=theano.config.floatX
        )

        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.output = relu((input.flatten(3) * self.W).sum(2) + self.b)

        self.params = [self.W, self.b]

    def save_state(self):
        return {
            "W": self.W.get_value(),
            "b": self.b.get_value(),
        }

    def load_state(self, state):
        self.W.set_value(state["W"])
        self.b.set_value(state["b"])


class Network(object):
    def __init__(self, rng,
                 batch_size,
                 n_kernels1,
                 n_kernels2,
                 n_kernels3,
                 pattern1_size,
                 pattern2_size,
                 pattern3_size,
                 pool2_size,
                 sequence_size):
        index = T.lscalar()  # index to a [mini]batch
        x = T.tensor4('x')  # the data is bunch of 3D patterns
        y = T.ivector('y')  # the labels are presented as 1D vector of
        is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction

        self.sequence_size = sequence_size
        self.batch_size = batch_size

        self.index = index
        self.x = x
        self.y = y
        self.is_train = is_train

        # Parameter of network


        # BUILD ACTUAL MODEL
        print '... building the model'

        # Construct the first convolutional pooling layer:
        conv_1, conv1_out_size = self.get_conv1_layer(rng, n_kernels1, pattern1_size, sequence_size)

        conv_1_output = add_dropout(conv_1.output, is_train, 0.8)
        conv_2 = LeNetConvPoolLayer(
            rng,
            input=conv_1_output,
            image_shape=(batch_size, n_kernels1, conv1_out_size, 1),
            filter_shape=(n_kernels2, n_kernels1, pattern2_size, 1),
            poolsize=(pool2_size, 1)
        )

        print("conv_2.W.shape={}".format(conv_2.W.get_value().shape))

        conv2_out_size = (conv1_out_size - pattern2_size + 1) / pool2_size
        conv_2_output = add_dropout(conv_2.output, is_train, 0.8)

        conv_3 = LeNetConvPoolLayer(
            rng,
            input=conv_2_output,
            image_shape=(batch_size, n_kernels2, conv2_out_size, 1),
            filter_shape=(n_kernels3, n_kernels2, pattern3_size, 1),
            poolsize=(2, 1)
        )

        print("conv_3.W.shape={}".format(conv_3.W.get_value().shape))

        conv3_out_size = (conv2_out_size - pattern3_size + 1) / 2
        conv_3_output = add_dropout(conv_3.output, is_train, 0.8)

        mr_layer = MultiRegressionLayer(rng, conv_3_output.flatten(3), n_kernels3, conv3_out_size)

        fully_connected = HiddenLayer(
            rng,
            add_dropout(mr_layer.output, is_train, 0.5),
            n_kernels3,
            n_kernels3,
            activation=relu)

        regression = LogisticRegression(input=fully_connected.output, n_in=n_kernels3, n_out=2)

        self.predict = theano.function(
            [x],
            regression.y_pred,
            givens={
                is_train: numpy.cast['int32'](0)
            }
        )

        self.prob = theano.function(
            [x],
            regression.p_y_given_x,
            givens={
                is_train: numpy.cast['int32'](0)
            }
        )

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.conv_3 = conv_3
        self.mr_layer = mr_layer
        self.fully_connected = fully_connected
        self.regression = regression

    def get_layers(self, x):
        result = (
            self.conv_1.output,
            self.conv_2.output,
            self.conv_3.output,
            self.mr_layer.output,
            self.fully_connected.output,
            self.regression.p_y_given_x)

        layers_fun = theano.function(
            [self.x],
            result
        )

        return layers_fun(x)

    def save_state(self):
        return {
            "conv_1": self.conv_1.save_state(),
            "conv_2": self.conv_2.save_state(),
            "conv_3": self.conv_3.save_state(),
            "mr_layer": self.mr_layer.save_state(),
            "fully_connected": self.fully_connected.save_state(),
            "regression": self.regression.save_state()
        }

    def load_state(self, state):
        self.conv_1.load_state(state["conv_1"])
        self.conv_2.load_state(state["conv_2"])
        self.conv_3.load_state(state["conv_3"])
        self.mr_layer.load_state(state["mr_layer"])
        self.fully_connected.load_state(state["fully_connected"])
        self.regression.load_state(state["regression"])

    def get_conv1_layer(self, rng, n_kernels1, pattern1_size, poolsize=(2, 1)):
        sequence_size = self.sequence_size
        conv_1 = LeNetConvPoolLayer(
            rng,
            input=self.x.reshape((self.batch_size, 4, sequence_size, 1)),
            image_shape=(self.batch_size, 4, sequence_size, 1),
            filter_shape=(n_kernels1, 4, pattern1_size, 1),
            poolsize=poolsize
        )
        print("conv_1.W.shape={}".format(conv_1.W.get_value().shape))
        conv1_out_size = (sequence_size - pattern1_size + 1) / 2
        return conv_1, conv1_out_size

class Fitter():
    def __init__(self,
                 network,
                 training,
                 validation,
                 test,
                 batch_size,
                 learning_rate,
                 reg_coef1,
                 reg_coef2):
        train_set_x, train_set_y = training
        validation_set_x, validation_set_y = validation
        test_set_x, test_set_y = test

        self.network = network
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        self.n_validation_batches = validation_set_x.get_value(borrow=True).shape[0] // batch_size

        x = network.x
        y = network.y
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
        cost = network.regression.negative_log_likelihood(y) + reg_coef1 * L1 + reg_coef2 * L2

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
            outputs=cost,
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
        patience = 50
        step = 1
        best_error = 1.0
        test_error = 0

        n_train_batches = self.n_train_batches

        log = open(log_path, 'w')
        for epoch in range(1000):
            a_cost = 0.0
            best_valid_error = 1.0
            epoch_start = time.time()
            for minibatch_index in xrange(n_train_batches):
                sys.stdout.write("*")
                sys.stdout.flush()
                avg_cost = self.train_model(minibatch_index)
                a_cost += avg_cost / n_train_batches
                if step % (n_train_batches // 5) == 0:
                    validation_error = self.get_validation_error()

                    if validation_error < best_valid_error:
                        best_valid_error = validation_error

                    if validation_error < best_error:
                        if model_path is not None:
                            with gzip.open(model_path, 'w') as f:
                                cPickle.dump(self.network.save_state(), f)
                        patience = 50
                        best_error = validation_error
                        test_error = self.get_test_error()
                    else:
                        patience -= 1
                step += 1

            print ""
            if patience < 0:
                break

            epoch_end = time.time()
            line = ("epoch: {}, cost: {:.5f}, valid err: {:.5f}, best err: {:.5f}, test err: {:.5f}, time: {}"
                    .format(epoch, a_cost, best_valid_error, best_error, test_error, human_time(epoch_end - epoch_start)))
            print(line)
            log.write(line + "\n")

            epoch += 1
        test_end = time.time()
        log.write("result: {}\n".format(test_error))
        log.write("time: {}\n".format(human_time(test_end - test_start)))
        log.close()
        return test_error


def create_default_network(sequence_size, batch_size=1000):
    rng = numpy.random.RandomState(23455)
    model = Network(rng,
                    batch_size,
                    n_kernels1=32,
                    n_kernels2=64,
                    n_kernels3=64,
                    pattern1_size=4,
                    pattern2_size=6,
                    pattern3_size=6,
                    pool2_size=2,
                    sequence_size=sequence_size)
    return model


def get_model_name(data_name, index):
    return "best_conv_model_{}_{}".format(data_name, index)


def get_model_parameters_path(dataset_name, index):
    model_name = get_model_name(dataset_name, index)
    model_path = 'models/{}.pkl.gz'.format(model_name)
    return model_path


def train_model(data, dataset_name, index, sequence_size=2000):
    training, validation, test = data
    batch_size = 1000
    network = create_default_network(sequence_size, batch_size)
    fitter = Fitter(network,
                    training,
                    validation,
                    test,
                    batch_size=batch_size,
                    learning_rate=0.001,
                    reg_coef1=0.00001,
                    reg_coef2=0.00001)
    model_name = get_model_name(dataset_name, index)
    log_path = 'models/{}.log'.format(model_name)
    model_path = get_model_parameters_path(dataset_name, index)
    fitter.do_fit(log_path, model_path)


def get_best_interval():
    return 1000, 2500


def get_dataset_types():
    return ["genes-coding", "genes-all", "cage-near-coding", "cage-all"]


def main():
    theano.config.openmp = True
    left, right = get_best_interval()

    for data_name in get_dataset_types():
        for i in xrange(3):
            data_set = divide_data(data_name, i)
            data = prepare_data(data_set, interval=(left, right))

            train_model(data, data_name, index=i, sequence_size=(right - left))


if __name__ == '__main__':
    main()
