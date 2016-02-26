import random
import sys
import numpy

import theano
import theano.tensor as T

from logistic import LogisticRegression
from theano_util import HiddenLayer
from data import shared_dataset, divide_data


def make_nmers(n):
    if n == 0:
        return [""]
    result = []
    for ch in ["a", "t", "g", "c"]:
        prev = make_nmers(n - 1)
        result.extend([ch + p for p in prev])
    return result


def convert_to_nmers(strings, n):
    nmers = make_nmers(n)
    nmers_map = {}

    for i, nmer in enumerate(nmers):
        nmers_map[nmer] = i

    result = []

    for s in strings:
        vector = [0] * (4 ** n)
        for start in range(len(s) - n):
            index = nmers_map[s[start:start + n]]
            vector[index] += 1
        result.append(vector)

    return result


def prepare_data(n):
    """ Load sequence data divide it to train and test set, and convert all to Theano shared.
    """
    train, valid, test = divide_data("genes-coding", 0)

    def unzip(l):
        return [a for a, b in l], [b for a, b in l]

    return shared_dataset(unzip(train)), unzip(test), unzip(valid)


def just_do_it():
    n = 6
    training, test, validation = prepare_data(n)
    print "Data prepared"

    train_set_x, train_set_y = training

    batch_size = 1000
    learning_rate = 0.003
    input_size = 4 ** n

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.ivector('y')



    rng = numpy.random.RandomState(23455)

    layer0 = HiddenLayer(
        rng,
        input=x,
        n_in=input_size,
        n_out=100)

    layer1 = LogisticRegression(input=layer0.output, n_in=100, n_out=2)

    L1 = abs(layer0.W).sum() + abs(layer1.W).sum()
    L2_sqr = (layer0.W ** 2).sum() + (layer1.W ** 2).sum()

    reg_coef = 0.001

    # the cost we minimize during training is the NLL of the model
    cost = layer1.negative_log_likelihood(y) + reg_coef * L1 + reg_coef * L2_sqr

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x, y],
        layer1.errors(y),
    )

    predict = theano.function(
         [x],
         layer1.y_pred
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
         for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    def get_error(data):
        avg_error = 0
        for i in range(5):
            avg_error += test_model(data[0][i * batch_size:(i + 1) * batch_size],
                                    data[1][i * batch_size:(i + 1) * batch_size])
        avg_error /= 5.0
        return avg_error

    patience = 20
    step = 1
    best_error = 100000
    result = 0
    for epoch in range(1000):
        a_cost = 0.0
        for minibatch_index in xrange(n_train_batches):
            sys.stdout.write("*")
            sys.stdout.flush()
            avg_cost = train_model(minibatch_index)
            a_cost += avg_cost
            if step % 10 == 0:
                test_error = get_error(test)

                if test_error < best_error:
                    patience = 20
                    best_error = test_error
                    result = get_error(validation)
                else:
                    patience -= 1
            step += 1

        print ""
        if patience < 0:
            break

        print("epoch:             " + str(epoch))
        print("cost:              " + str(a_cost / n_train_batches))
        print("Best error:        " + str(best_error))
        print("Validation error:  " + str(result))
        epoch += 1

    print "result: {}".format(result)


if __name__ == '__main__':
    just_do_it()

