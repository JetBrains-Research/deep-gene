import theano
import numpy
import random
import os
import gzip
import cPickle


def convert_to_binary_layered(s):
    letters = {
        'a': [1, 0, 0, 0],
        't': [0, 1, 0, 0],
        'g': [0, 0, 1, 0],
        'c': [0, 0, 0, 1]}

    l = numpy.zeros(shape=(4, len(s), 1), dtype=numpy.float32)

    for i, letter in enumerate(s):
        l[:, i, 0] = letters[letter]

    return l


def convert_to_binary_flat(s):
    letters = {
        'a': [1, 0, 0, 0],
        't': [0, 1, 0, 0],
        'g': [0, 0, 1, 0],
        'c': [0, 0, 0, 1]}

    l = []

    for letter in s:
        l.append(letters[letter])

    return l


def divide_data(name, index):
    """ Load sequence data divide it to train and test set.
    """
    path = "data/divided_{}_{}".format(name, index) + ".pkl.gz"
    if not os.path.exists(path):
        def sum_size(l):
            return sum(map(len, l))

        def cut(l, size):
            acc = 0
            i = 0
            for i, e in enumerate(l):
                acc += len(e)
                if acc >= size: break

            return i + 1

        with open("data/{}.txt".format(name)) as f:
            pos = [line.split("\t") for line in f.readlines()]
        with open("data/negative.txt") as f:
            neg = [line.split("\t") for line in f.readlines()]

        pos_size = sum_size(pos)
        random.shuffle(neg)
        neg = neg[:cut(neg, pos_size)]
        neg_size = sum_size(neg)

        combined = zip(pos, [1] * len(pos)) + zip(neg, [0] * len(neg))

        random.shuffle(combined)

        print "Data loaded"
        print("pos {}, neg  {}".format(pos_size, neg_size))

        def divide(data, size):
            c = cut([it[0] for it in data], size)
            return data[:c], data[c:]

        test_size = ((pos_size + neg_size) // 10000) * 1000
        test, combined = divide(combined, test_size)
        valid, combined = divide(combined, test_size)
        train = combined

        def flatten_first(data):
            result = []
            for (seqs, y) in data:
                for e in seqs:
                    result.append((e, y))
            return result

        result = (flatten_first(train), flatten_first(valid), flatten_first(test))

        print("train: {}".format(len(result[0])))
        print("valid: {}".format(len(result[1])))
        print("test: {}".format(len(result[2])))

        with gzip.open(path, 'w') as f:
            cPickle.dump(result, f)

        return result

    with gzip.open(path, 'r') as f:
        return cPickle.load(f)


def unzip(l):
        return [a for a, b in l], [b for a, b in l]


def human_time(t):
    seconds = int(t)
    milliseconds = t - int(t)

    hours = seconds // (60 * 60)
    minutes = (seconds // 60) % 60
    seconds %= 60

    result = ""
    if hours > 0:
        result += str(hours) + "h"
    if minutes > 0:
        result += str(minutes) + "m"

    if minutes >= 10 or hours > 0:
        result += str(seconds) + "s"
    else:
        result += "{:2.2f}s".format(seconds + milliseconds)
    return result


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype='int32'),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y