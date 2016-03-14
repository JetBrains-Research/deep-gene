import gzip
import cPickle
import itertools

import numpy
from conv import get_best_interval, get_pattern_function1, get_pattern_function2, create_default_network
import matplotlib.pyplot as plt

from data import convert_to_binary_layered, divide_data, unzip


def make_nmers(n):
    return ["".join(chunk) for chunk in itertools.product("atgc", repeat=n)]


def inspect_pattern1(kernel1, conv_1):
    print "Write first patterns"
    all_nmers = make_nmers(4)
    fun = get_pattern_function1(
        batch_size=len(all_nmers),
        sequence_size=4,
        pattern1_size=4,
        n_kernels1=kernel1,
        conv_1_params=conv_1
    )
    results = fun([convert_to_binary_layered(s) for s in all_nmers])

    with open('patterns/conv_1_pattern.txt', 'w') as f:
        for i in xrange(0, kernel1):
            f.write("\nPattern: {}\n".format(i))
            t = results[:, i, 0, 0]
            zipped = zip(t.tolist(), all_nmers)
            for result, nmer in sorted(zipped):
                f.write("{} {}\n".format(nmer, result))

    print "Write first patterns. end"


def cut_to_nmers(seqs, k):
    nmers_set = set()
    for seq in seqs:
        for i in xrange(len(seq) - k):
            nmers_set.add(seq[i:i + k])
    return nmers_set


def get_nmers(n):
    train, valid, test = divide_data("genes-coding", 0)
    nmers_set = set()
    nmers_set = nmers_set.union(cut_to_nmers(unzip(train)[0], n))
    nmers_set = nmers_set.union(cut_to_nmers(unzip(valid)[0], n))
    nmers_set = nmers_set.union(cut_to_nmers(unzip(test)[0], n))
    return list(nmers_set)


def inspect_pattern2(conv_1, conv_2, kernel1, kernel2, w):
    n = 15
    nmers_list = get_nmers(n)
    print("{}-mer set: {}".format(n, len(nmers_list)))
    best = [[] for _ in xrange(kernel2)]
    fun = get_pattern_function2(
        batch_size=1000,
        sequence_size=15,
        pattern1_size=4,
        pattern2_size=6,
        n_kernels1=kernel1,
        n_kernels2=kernel2,
        layer0_params=conv_1,
        layer1_params=conv_2)

    for offset in range(0, len(nmers_list), 1000):
        print("{}/{}".format(offset, len(nmers_list)))

        nmers = nmers_list[offset:offset + 1000]
        results = fun([convert_to_binary_layered(s) for s in nmers])

        for i in xrange(kernel2):
            t = results[:, i, 0, 0]
            best[i].extend(zip(t.tolist(), nmers))
            best[i] = list(sorted(best[i]))[-1000:]


        for i in xrange(kernel2):
            with open('patterns/conv_2_pattern_{}.txt'.format(i), 'w') as f:
                f.write("Pattern {} best\n".format(i))
                for score, seq in best[i]:
                    f.write(seq + " " + str(score) + "\n")


def do_it():
    with gzip.open('models/best_conv_model_genes-coding_0.pkl.gz', 'r') as f:
        state = cPickle.load(f)

    conv_1 = state["conv_1"]
    conv_2 = state["conv_2"]
    mr_layer = state["mr_layer"]

    kernel1 = 32
    kernel2 = 64

    w = mr_layer["W"]

    inspect_pattern1(kernel1, conv_1)

    for i in xrange(kernel2):
        x = numpy.arange(0, w.shape[1])
        plt.plot(x, w[i, :], color="b")
        plt.savefig('patterns/plot_{}.png'.format(i))
        plt.close()

    inspect_pattern2(conv_1, conv_2, kernel1, kernel2, w)


if __name__ == '__main__':
    do_it()
