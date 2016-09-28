import os
import sys
import gzip
import cPickle

import numpy
import theano
import theano.tensor as T
import time

from util.adam import adam
from util.logistic import LogisticRegression
from util.theano_util import LeNetConvPoolLayer, HiddenLayer, add_dropout, relu
from util.data import divide_data, shared_dataset, unzip, human_time, convert_to_number


def prepare_data(data, interval):
    left, right = interval
    train, valid, test = data

    def prepossess(d):
        binary_data = []
        for (s, t) in d:
            binary_data.append((convert_to_number(s[left:right]), t))
        return shared_dataset(unzip(binary_data))

    return prepossess(train), prepossess(valid), prepossess(test)


def create_conv_input(x, batch_size, sequence_size):
    l_a = T.eq(x, 0).reshape((batch_size, sequence_size, 1))
    l_t = T.eq(x, 1).reshape((batch_size, sequence_size, 1))
    l_g = T.eq(x, 2).reshape((batch_size, sequence_size, 1))
    l_c = T.eq(x, 3).reshape((batch_size, sequence_size, 1))
    return T.cast(T.stack([l_a, l_t, l_g, l_c], axis=1), theano.config.floatX)

class MultiRegressionLayer_old(object):
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

class ConvolutionPart2(object):
    def __init__(self, rng,
                 parameters,
                 batch_size,
                 sequence_size,
                 conv_input,
                 is_train,
                 inspect=False):

        n_kernels1 = parameters["n_kernels1"]
        n_kernels2 = parameters["n_kernels2"]
        pattern1_size = parameters["pattern1_size"]
        pattern2_size = parameters["pattern2_size"]
        dropout1 = parameters["dropout1"]

        conv_1 = LeNetConvPoolLayer(
            rng,
            input=conv_input,
            image_shape=(batch_size, 4, sequence_size, 1),
            filter_shape=(n_kernels1, 4, pattern1_size, 1),
            poolsize=(2, 1)
        )
        conv1_out_size = (sequence_size - pattern1_size + 1) // 2
        conv_1_output = add_dropout(conv_1.output, is_train, 1 - dropout1, rng)

        conv_2 = LeNetConvPoolLayer(
            rng,
            input=conv_1_output,
            image_shape=(batch_size, n_kernels1, conv1_out_size, 1),
            filter_shape=(n_kernels2, n_kernels1, pattern2_size, 1),
            poolsize=((1, 1) if inspect else (2, 1))
        )

        conv2_out_size = (conv1_out_size - pattern2_size + 1) // (1 if inspect else 2)

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.conv2_out_size = conv2_out_size

    def load_state(self, state):
        self.conv_1.load_state(state["conv_1"])
        self.conv_2.load_state(state["conv_2"])


class ConvolutionPart3(object):
    def __init__(self, rng,
                 parameters,
                 batch_size,
                 sequence_size,
                 conv_input,
                 is_train,
                 inspect=False):

        n_kernels2 = parameters["n_kernels2"]
        n_kernels3 = parameters["n_kernels3"]
        pattern3_size = parameters["pattern3_size"]
        dropout2 = parameters["dropout2"]

        convolution_part = ConvolutionPart2(
            rng,
            parameters,
            batch_size,
            sequence_size,
            conv_input,
            is_train)

        conv_1 = convolution_part.conv_1
        conv_2 = convolution_part.conv_2
        conv2_out_size = convolution_part.conv2_out_size

        conv_2_output = add_dropout(conv_2.output, is_train, 1 - dropout2, rng)

        conv_3 = LeNetConvPoolLayer(
            rng,
            input=conv_2_output,
            image_shape=(batch_size, n_kernels2, conv2_out_size, 1),
            filter_shape=(n_kernels3, n_kernels2, pattern3_size, 1),
            poolsize=((1, 1) if inspect else (2, 1))
        )

        conv3_out_size = (conv2_out_size - pattern3_size + 1) // (1 if inspect else 2)

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.conv_3 = conv_3
        self.conv3_out_size = conv3_out_size

    def load_state(self, state):
        self.conv_1.load_state(state["conv_1"])
        self.conv_2.load_state(state["conv_2"])
        self.conv_3.load_state(state["conv_3"])


class Network(object):
    def __init__(self, rng,
                 batch_size,
                 parameters):

        sequence_size = parameters["right"] - parameters["left"]
        n_kernels3 = parameters["n_kernels3"]
        n_fully_connected = parameters["n_fully_connected"]
        dropout0 = parameters["dropout0"]
        dropout3 = parameters["dropout3"]
        dropout4 = parameters["dropout4"]
        dropout5 = parameters["dropout5"]

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x', dtype='int8')  # the data is bunch of 3D patterns
        is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction

        self.sequence_size = sequence_size
        self.batch_size = batch_size

        self.index = index
        self.x = x
        self.is_train = is_train

        # BUILD ACTUAL MODEL
        print '... building the model'
        conv_input = add_dropout(create_conv_input(x, batch_size, sequence_size), is_train, 1 - dropout0, rng)

        self.conv_input = conv_input

        convolution_part = ConvolutionPart3(
            rng,
            parameters,
            batch_size,
            sequence_size,
            conv_input,
            is_train)

        conv_1 = convolution_part.conv_1
        conv_2 = convolution_part.conv_2
        conv_3 = convolution_part.conv_3
        conv3_out_size = convolution_part.conv3_out_size

        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.conv_3 = conv_3

        conv_3_output = add_dropout(conv_3.output, is_train, 1 - dropout3, rng)

        mr_layer = MultiRegressionLayer_old(rng, conv_3_output.flatten(3), n_kernels3, conv3_out_size)

        fully_connected = HiddenLayer(
            rng,
            add_dropout(mr_layer.output, is_train, 1-dropout4, rng),
            n_kernels3,
            n_fully_connected,
            activation=relu)

        regression_input = add_dropout(fully_connected.output, is_train, 1 - dropout5, rng)
        regression = LogisticRegression(input=regression_input, n_in=n_fully_connected, n_out=2)

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


def create_default_network(batch_size=1000):
    rng = numpy.random.RandomState(23455)
    model = Network(rng, batch_size, get_default_parameters())
    return model


def get_model_name(data_name, index):
    return "best_conv_model_{}_{}".format(data_name, index)


def get_model_parameters_path(dataset_name, index):
    model_name = get_model_name(dataset_name, index)
    model_path = 'models/{}.pkl.gz'.format(model_name)
    return model_path


def train_model(data, dataset_name, index):
    training, validation, test = data
    batch_size = 1000
    network = create_default_network(batch_size)
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
        "reg_coef1": 0.00001,
        "reg_coef2": 0.00001,
    }


def get_dataset_types_mm9():
    return ["mm9_genes_coding", "mm9_genes_all", "mm9_cage_near_coding", "mm9_cage_all"]

def get_dataset_types_hg19():
    return ["hg19_genes_coding", "hg19_genes_all", "hg19_cage_near_coding", "hg19_cage_all"]

def get_dataset_types():
    return get_dataset_types_mm9() + get_dataset_types_hg19()


def main():
    theano.config.openmp = True
    left, right = get_best_interval()

    for data_name in get_dataset_types():
        for i in xrange(3):
            path = get_model_parameters_path(data_name, i)
            if os.path.exists(path):
                print "Model {} exists.".format(path)
                continue
            data_set = divide_data(data_name, i)
            data = prepare_data(data_set, interval=(left, right))

            train_model(data, data_name, index=i)


if __name__ == '__main__':
    main()
