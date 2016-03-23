import gzip
import cPickle
import itertools
import theano
import theano.tensor as T

import numpy
from conv import get_default_parameters, get_model_parameters_path, ConvolutionPart
import matplotlib.pyplot as plt

from data import convert_to_binary_layered, divide_data, unzip


def cut_to_nmers(seqs, k):
    for seq in seqs:
        for i in xrange(len(seq) - k):
            yield seq[i:i + k]


def get_nmers_iterator(n):
    train, valid, test = divide_data("genes-coding", 0)
    return itertools.chain(cut_to_nmers(unzip(train)[0], n),
                           cut_to_nmers(unzip(valid)[0], n),
                           cut_to_nmers(unzip(test)[0], n))

def get_nmers(n):
    nmers_set = set()
    for nmer in get_nmers_iterator(n):
        nmers_set.add(nmer)
        if len(nmers_set) >= 1000 * 1000:
            break
    return list(nmers_set)


def inspect_pattern(model_path):
    n = 37

    default_parameters = get_default_parameters()

    kernel3 = default_parameters["n_kernels3"]

    batch_size = 1000

    # for i in xrange(kernel2):
    #plt.plot(x, w[i, :], color="b")
    #plt.savefig('patterns/plot_{}.png'.format(i))
    #plt.close()

    rng = numpy.random.RandomState(23455)

    x = T.tensor4('x')  # the data is bunch of 3D patterns
    is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction

    convolution = ConvolutionPart(
        rng,
        default_parameters,
        batch_size,
        37,
        x,
        is_train,
        inspect=True)

    with gzip.open(model_path, 'r') as f:
        loaded_state = cPickle.load(f)
        convolution.load_state(loaded_state)

    predict = theano.function(
        [x],
        convolution.conv_3.output,
        givens={
            is_train: numpy.cast['int32'](0)
        }
    )

    nmers_list = get_nmers(n)

    print("{}-mer set: {}".format(n, len(nmers_list)))
    best = [[] for _ in xrange(kernel3)]

    for offset in range(0, len(nmers_list), 1000):
        print("{}/{}".format(offset, len(nmers_list)))

        nmers = nmers_list[offset:offset + 1000]
        results = predict([convert_to_binary_layered(s) for s in nmers])

        for i in xrange(kernel3):
            t = results[:, i, 0, 0]
            best[i].extend(zip(t.tolist(), nmers))
            best[i] = list(sorted(best[i]))[-1000:]


        for i in xrange(kernel3):
            with open('patterns/conv_2_pattern_{}.txt'.format(i), 'w') as f:
                f.write("Pattern {} best\n".format(i))
                for score, seq in best[i]:
                    f.write(seq + " " + str(score) + "\n")


def do_it():
    inspect_pattern(get_model_parameters_path("genes-coding", 0))


if __name__ == '__main__':
    do_it()
