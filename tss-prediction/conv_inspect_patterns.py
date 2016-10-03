import cPickle
import gzip
import itertools
import os
import subprocess

import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as T
from conv import get_default_parameters, get_model_parameters_path, ConvolutionPart3, ConvolutionPart2, \
    get_dataset_types, create_conv_input

from util.data import divide_data, unzip, convert_to_number


def cut_to_nmers(seqs, k):
    for seq in seqs:
        for i in xrange(len(seq) - k):
            yield seq[i:i + k]


def get_nmers_iterator(n):
    train, valid, test = divide_data("mm9_genes_coding", 0)
    return itertools.chain(cut_to_nmers(unzip(train)[0], n),
                           cut_to_nmers(unzip(valid)[0], n),
                           cut_to_nmers(unzip(test)[0], n))


def write_paterns(path, best, n_kernels, prefix):
    for i in xrange(n_kernels):
        base_name = path + '/{}_pattern_{}'.format(prefix, i)
        with open(base_name + '.txt', 'w') as f:
            f.write("Pattern {} best\n".format(i))
            for score, seq in best[i]:
                f.write(seq + " " + str(score) + "\n")

        with open(base_name + '.fasta', 'w') as f:
            for score, seq in best[i]:
                f.write(seq + "\n")

        subprocess.call(["weblogo",
                         "-s", "large",
                         "-c", "classic",
                         "-F", "png",
                         "-f", base_name + ".fasta",
                         "-o", base_name + ".png"])


def inspect_patterns(prefix, batch_size, n_kernels, n, predict, path):
    best = [[] for _ in xrange(n_kernels)]
    used_nmers = set()
    nmers = []
    for nmer in get_nmers_iterator(n):
        if nmer in used_nmers:
            continue

        nmers.append(nmer)
        used_nmers.add(nmer)
        if len(nmers) == batch_size:
            results = predict([convert_to_number(s) for s in nmers])

            for i in xrange(n_kernels):
                t = results[:, i, 0, 0]
                best[i].extend(zip(t.tolist(), nmers))
                best[i] = list(sorted(best[i]))[-500:]

            nmers = []

        if len(used_nmers) > 2000000:
            used_nmers = set()
            print("Clean cache")

    write_paterns(path, best, n_kernels, prefix)


def inspect_pattern3(model_path, path):
    n = 37

    default_parameters = get_default_parameters()

    kernel3 = default_parameters["n_kernels3"]

    batch_size = 1000

    rng = numpy.random.RandomState(23455)

    x = T.matrix('x', dtype='int8')
    is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction

    conv_input = create_conv_input(x, batch_size, n)

    convolution = ConvolutionPart3(
        rng,
        default_parameters,
        batch_size,
        n,
        conv_input,
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

    inspect_patterns("conv3", batch_size, kernel3, n, predict, path)


def inspect_pattern2(model_path, path):
    n = 15

    default_parameters = get_default_parameters()

    kernel2 = default_parameters["n_kernels2"]

    batch_size = 1000

    rng = numpy.random.RandomState(23455)

    x = T.matrix('x', dtype='int8')
    is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction

    conv_input = create_conv_input(x, batch_size, n)

    convolution = ConvolutionPart2(
        rng,
        default_parameters,
        batch_size,
        n,
        conv_input,
        is_train,
        inspect=True)

    with gzip.open(model_path, 'r') as f:
        loaded_state = cPickle.load(f)
        convolution.load_state(loaded_state)

    predict = theano.function(
        [x],
        convolution.conv_2.output,
        givens={
            is_train: numpy.cast['int32'](0)
        }
    )

    inspect_patterns("conv2", batch_size, kernel2, n, predict, path)


def get_patterns(data_set):
    path = "patterns/" + data_set
    os.mkdir(path)
    default_parameters = get_default_parameters()
    model_path = get_model_parameters_path(data_set, 0)
    kernel3 = default_parameters["n_kernels3"]
    with gzip.open(model_path, 'r') as f:
        loaded_state = cPickle.load(f)
        mr_layer = loaded_state["mr_layer"]
    w = mr_layer["W"]
    for i in xrange(kernel3):
        x = numpy.arange(0, w.shape[1])
        plt.plot(x, w[i, :], color="b")
        plt.savefig(path + '/plot_{}.png'.format(i))
        plt.close()
    inspect_pattern2(model_path, path)
    inspect_pattern3(model_path, path)


def do_it():
    for data_set in get_dataset_types():
        get_patterns(data_set)


if __name__ == '__main__':
    do_it()
