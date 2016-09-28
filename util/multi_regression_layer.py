import theano
import numpy
import lasagne.init
from lasagne import nonlinearities
from lasagne.layers.base import Layer


class MultiRegressionLayer(Layer):
    def __init__(self, incoming,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(MultiRegressionLayer, self).__init__(incoming, **kwargs)

        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = self.input_shape[1]
        num_input = self.input_shape[2]

        self.W = self.add_param(W, (self.num_units, num_input), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_units,), name="b",
                                    regularizable=False)




    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units

    def get_output_for(self, input, **kwargs):

        activation = (input * self.W).sum(2)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)