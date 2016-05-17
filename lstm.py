import random
import numpy
import gzip
import cPickle

import theano
import theano.tensor as T
from adam import adam

from data import divide_data, convert_to_binary_flat
from logistic import LogisticRegression


def prepare_data(data, interval):
    left, right = interval
    train, test, valid = data

    def prepossess(d):
        binary_data = []
        for (s, t) in d:
            binary_data.append((convert_to_binary_flat(s[left:right]), t))
        return [a for a, b in binary_data], [b for a, b in binary_data]

    return prepossess(train), prepossess(test), prepossess(valid)


def make_random_matrix(name, a, b):
    bound = numpy.sqrt(1. / (a + b))
    return theano.shared(name=name,
                         value=numpy.asarray(
                             numpy.random.uniform(-bound, bound, (a, b)),
                             dtype=theano.config.floatX
                         ))


def make_zeros_vector(name, n):
    return theano.shared(name=name,
                         value=numpy.zeros(n, dtype=theano.config.floatX))


class LSTM(object):
    ''' elman neural net model '''

    def __init__(self, n_hidden, x_dimension, n_classes, sequence_len, batch_size, learning_rate=0.01):
        """
        n_hidden :: dimension of the hidden layer
        n_classes :: number of classes
        x_dimension :: dimension of x
        """

        params_map = {
            "w_i": make_random_matrix('w_i', x_dimension, n_hidden),
            "u_i": make_random_matrix('u_i', n_hidden, n_hidden),
            "b_i": make_zeros_vector('b_i', n_hidden),

            "w_f": make_random_matrix('w_f', x_dimension, n_hidden),
            "u_f": make_random_matrix('u_f', n_hidden, n_hidden),
            "b_f": make_zeros_vector('b_f', n_hidden),

            "w_c": make_random_matrix('w_c', x_dimension, n_hidden),
            "u_c": make_random_matrix('u_c', n_hidden, n_hidden),
            "b_c": make_zeros_vector('b_c', n_hidden),

            "w_o": make_random_matrix('w_o', x_dimension, n_hidden),
            "u_o": make_random_matrix('u_o', n_hidden, n_hidden),
            "v_o": make_random_matrix('v_o', n_hidden, n_hidden),
            "b_o": make_zeros_vector('b_o', n_hidden),

            "u_s": make_random_matrix('u_s', n_hidden, n_classes),
            "v_s": make_random_matrix('v_s', n_hidden, n_classes),
            "b_s": make_zeros_vector('b_s', n_classes),

            "h0": theano.shared(name='h0',
                                value=numpy.zeros((batch_size, n_hidden),
                                                  dtype=theano.config.floatX)),
            "c0": theano.shared(name='h0',
                                value=numpy.zeros((batch_size, n_hidden),
                                                  dtype=theano.config.floatX))
        }

        self.params_map = params_map

        x = T.tensor3()
        y = T.ivector('y')  # labels

        w_i = params_map["w_i"]
        u_i = params_map["u_i"]
        b_i = params_map["b_i"]

        w_f = params_map["w_f"]
        u_f = params_map["u_f"]
        b_f = params_map["b_f"]

        w_c = params_map["w_c"]
        u_c = params_map["u_c"]
        b_c = params_map["b_c"]

        w_o = params_map["w_o"]
        u_o = params_map["u_o"]
        v_o = params_map["v_o"]
        b_o = params_map["b_o"]

        u_s = params_map["u_s"]
        v_s = params_map["v_s"]
        b_s = params_map["b_s"]

        h0 = params_map["h0"]
        c0 = params_map["c0"]

        def recurrence(x_t, h_prev, c_prev):
            i_t = T.nnet.sigmoid(T.dot(x_t, w_i) + T.dot(h_prev, u_i) + b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, w_f) + T.dot(h_prev, u_f) + b_f)
            c_t_ = T.nnet.sigmoid(T.dot(x_t, w_c) + T.dot(h_prev, u_c) + b_c)

            c_t = i_t * c_t_ + f_t * c_prev

            o_t = T.nnet.sigmoid(T.dot(x_t, w_o) + T.dot(h_prev, u_o) + T.dot(c_t, v_o) + b_o)

            h_t = o_t * T.tanh(c_t)

            s_t = T.nnet.softmax(T.dot(h_t, u_s) + T.dot(c_t, v_s) + b_s)
            return [h_t, c_t, s_t]

        [h, c, s], _ = theano.scan(fn=recurrence,
                                   sequences=x,
                                   outputs_info=[h0, c0, None],
                                   n_steps=x.shape[0])

        input = s.dimshuffle((1, 0, 2)).flatten(2)
        layer1 = LogisticRegression(input=input, n_in=sequence_len * n_classes, n_out=2)

        self.layer1 = layer1

        # bundle
        self.params = list(params_map.values()) + layer1.params

        """
        L2 = None
        
        for name, value in self.params_map.iteritems():
            if "b_" in name:
                print name
                continue
            if L2:
                L2 = L2 + (value ** 2).sum()
            else:
                L2 = (value ** 2).sum()
        """

        cost = layer1.negative_log_likelihood(y)

        updates = adam(cost, self.params, lr=learning_rate)

        # theano functions to compile

        self.train = theano.function(inputs=[x, y],
                                     outputs=cost,
                                     updates=updates)

        self.test_model = theano.function(
            [x, y],
            layer1.errors(y),
        )

    def save_state(self):
        result = {
            "layer1": self.layer1
        }
        for name, value in self.params_map.iteritems():
            result[name] = value.get_value()
        return result


def get_error(batch_size, lstm, data, sequence_len):
    data_x, data_y = data
    avg_error = 0
    n_batches = 0
    for i in xrange(0, len(data_x) - batch_size + 1, batch_size):
            block_x = [[] for _ in range(sequence_len)]
            for k in range(batch_size):
                for j in range(sequence_len):
                    block_x[j].append(data_x[i + k][j])

            block_y = []
            for k in range(batch_size):
                block_y.append(data_y[i + k])

            x = numpy.array(block_x, dtype=theano.config.floatX)
            y = numpy.array(block_y, dtype='int32')

            avg_error += lstm.test_model(x, y)
            n_batches += 1

    avg_error /= n_batches
    return avg_error


def shuffle_training(training):
    train_x, train_y = training
    indexes = range(len(train_x))
    random.shuffle(indexes)
    train_x = [train_x[i] for i in indexes]
    train_y = [train_y[i] for i in indexes]
    return train_x, train_y


def fit_model(data_set, index):
    training, test, validation = prepare_data(data_set, interval=(500, 2500))
    print "Data prepared"
    sequence_len = 2000
    batch_size = 200
    print "Build model"
    lstm = LSTM(200, 4, 30, sequence_len, batch_size, learning_rate=0.0001)
    print "Build model. DONE."
    patience = 20
    step = 1
    best_error = 100000
    result = 0
    random.seed(123)
    for epoch in range(1000):
        train_x, train_y = shuffle_training(training)

        print "epoch {}".format(epoch)

        avg_cost = 0.0
        n_train_batches = 0
        for i in xrange(0, len(train_x) - batch_size, batch_size):
            block_x = [[] for _ in range(sequence_len)]
            for k in range(batch_size):
                for j in range(sequence_len):
                    block_x[j].append(train_x[i + k][j])

            block_y = []
            for k in range(batch_size):
                block_y.append(train_y[i + k])

            x = numpy.array(block_x, dtype=theano.config.floatX)
            y = numpy.array(block_y, dtype='int32')

            cost = lstm.train(x, y)

            print "{} {}/{}, cost: {}".format(epoch, i, len(train_x), cost)

            avg_cost += cost
            n_train_batches += 1
            if step % 50 == 0:
                validation_error = get_error(batch_size, lstm, validation, sequence_len)

                if validation_error < best_error:
                    patience = 20
                    best_error = validation_error
                    result = get_error(batch_size, lstm, test, sequence_len)
                    with gzip.open('models/best_lstm_model_{}.pkl.gz'.format(index), 'w') as f:
                        cPickle.dump(lstm.save_state(), f)
                    print("Best error:        " + str(best_error))
                    print("Test error:        " + str(result))
                else:
                    print("Test error         " + str(validation_error))
                    print("Patience           " + str(patience))
                    patience -= 1
            step += 1

        print ""
        if patience < 0:
            break

        print("epoch:             " + str(epoch))
        print("cost:              " + str(avg_cost / n_train_batches))
        epoch += 1

    print "result: {}".format(result)
    return result


def just_do_it():
    data_sets = [divide_data("mm9_genes_coding", i) for i in range(0, 3)]

    results = []

    for i in xrange(3):
        result = fit_model(data_sets[i], i)
        results.append(result)

    for result in results:
        print "result: {}".format(result)

    print "Mean result: {}".format(sum(results) / 3.0)

if __name__ == '__main__':
    just_do_it()

