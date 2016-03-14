import gzip
import cPickle

import numpy
import theano
from conv import get_best_interval, create_network

from data import convert_to_binary_layered


def prepare_genome(interval):
    left, right = interval

    with open('data/chr1.fa', 'r') as f:
        genome = "".join(map(lambda t: t[:-1], f.readlines()[1:]))
        genome = genome.lower()

    for center in xrange(2000, len(genome) - 2000, 20):
        part = genome[center + left - 2000: center + right - 2000]
        if "n" not in part:
            yield part, center


def predict_tracks_conv():
    interval = get_best_interval()
    left, right = interval
    batch_size = 1000
    model = create_network(right - left, batch_size)
    for i in range(3):
        with gzip.open("models/best_conv_model_{}.pkl.gz".format(i), 'r') as f:
            loaded_state = cPickle.load(f)
            model.load_state(loaded_state)

        f = open("tracks/conv_predictions_{}.wig".format(i), 'w')
        f.write("variableStep chrom=chr1 span=20\n")
        parts = []
        offsets = []

        for (part, offset) in prepare_genome(interval):
            binary = convert_to_binary_layered(part)
            parts.append(binary)
            offsets.append(offset)
            if len(parts) == batch_size:
                data_x = numpy.asarray(parts, dtype=theano.config.floatX)
                results = model.prob(data_x)
                for i in xrange(0, batch_size):
                    f.write(str(offsets[i] - 10) + " " + str(results[i, 1]) + "\n")

                parts = []
                offsets = []

        f.close()


def do_it():
    """ Predict TSS along chromosome
    """
    theano.config.openmp = True

    predict_tracks_conv()

if __name__ == '__main__':
    do_it()