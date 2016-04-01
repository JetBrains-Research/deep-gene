import gzip
import cPickle

import numpy
import theano
from conv import get_best_interval, create_default_network, get_model_parameters_path, get_model_name, \
    get_dataset_types_mm9

from data import convert_to_number


def prepare_genome(interval):
    left, right = interval

    with open('data/chr1.fa', 'r') as f:
        genome = "".join(map(lambda t: t[:-1], f.readlines()[1:]))
        genome = genome.lower()

    for center in xrange(2000, len(genome) - 2000, 10):
        part = genome[center + left - 2000: center + right - 2000]
        if "n" not in part:
            yield part, center


def predict_tracks_conv():
    interval = get_best_interval()
    batch_size = 1000
    network = create_default_network(batch_size)
    for data_name in get_dataset_types_mm9():
        for i in range(3):
            model_name = get_model_name(data_name, i)
            with gzip.open(get_model_parameters_path(data_name, i), 'r') as f:
                loaded_state = cPickle.load(f)
                network.load_state(loaded_state)

            f = open("tracks/conv_predictions_{}.wig".format(model_name), 'w')
            f.write("variableStep chrom=chr1 span=10\n")
            parts = []
            offsets = []

            for (part, offset) in prepare_genome(interval):
                binary = convert_to_number(part)
                parts.append(binary)
                offsets.append(offset)
                if len(parts) == batch_size:
                    data_x = numpy.asarray(parts, dtype="int8")
                    results = network.prob(data_x)
                    for i in xrange(0, batch_size):
                        f.write(str(offsets[i]) + " " + str(results[i, 1]) + "\n")

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