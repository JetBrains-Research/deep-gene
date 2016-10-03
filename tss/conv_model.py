import numpy
import theano

import theano.tensor as T

from util.theano_util import relu
from util.adam import adam
from util.logistic import LogisticRegression
from util.theano_util import LeNetConvPoolLayer, HiddenLayer, add_dropout, relu
from util.data import divide_data, shared_dataset, unzip, human_time, convert_to_number


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
