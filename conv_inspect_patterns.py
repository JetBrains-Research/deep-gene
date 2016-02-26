import random
import sys
import gzip
import cPickle

import numpy
from conv import get_best_interval, get_pattern_function1, get_pattern_function2
import matplotlib.pyplot as plt

from data import convert_to_binary_layered, divide_data, unzip
from nmer import make_nmers


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


def make_random_nmers(n, number):
    result = []
    for i in xrange(number):
        str = ""
        for j in xrange(n):
            ch = random.choice(["a", "t", "g", "c"])
            str += ch
        result.append(str)

    return result


def do_it():
    with gzip.open('models/best_conv_model_0.pkl.gz', 'r') as f:
        state = cPickle.load(f)

    conv_1 = state["conv_1"]
    conv_2 = state["conv_2"]
    mr_layer = state["mr_layer"]

    kernel1 = 32
    kernel2 = 64

    w = mr_layer["W"]

    for i in xrange(kernel2):
        x = numpy.arange(0, w.shape[1])
        plt.plot(x, w[i, :], color="b")
        plt.savefig('patterns/plot_{}.png'.format(i))
        plt.close()

    inspect_pattern1(kernel1, conv_1)

    train, valid, test = divide_data("genes-coding", 0)
    seqs = unzip(train)[0]

    best = [[] for _ in xrange(kernel2)]

    worst = [[] for _ in xrange(kernel2)]

    for k in xrange(len(seqs)):
        print("{}/{}".format(k, len(seqs)))
        seq = seqs[k]
        nmers_set = set()
        for i in xrange(len(seq) - 15):
            nmers_set.add(seq[i:i + 15])

        nmers = list(nmers_set)
        fun = get_pattern_function2(
            batch_size=len(nmers),
            sequence_size=15,
            pattern1_size=4,
            pattern2_size=6,
            n_kernels1=kernel1,
            n_kernels2=kernel2,
            layer0_params=conv_1,
            layer1_params=conv_2
        )
        results = fun([convert_to_binary_layered(s) for s in nmers])

        for i in xrange(kernel2):
            t = results[:, i, 0, 0]
            best[i].extend(zip(t.tolist(), nmers))
            best[i] = list(sorted(best[i]))[-500:]

            worst[i].extend(zip(t.tolist(), nmers))
            worst[i] = list(sorted(worst[i]))[:500]

    for i in xrange(kernel2):
         with open('patterns/pattern_{}.txt'.format(i), 'w') as f:
            f.write("Pattern {}\n".format(i))
            f.write(str(w[i, :].tolist()) + "\n")
            f.write("best\n")
            for score, seq in best[i]:
                f.write(seq + " " + str(score) + "\n")
            f.write("worst\n")
            for score, seq in worst[i]:
                f.write(seq + " " + str(score) + "\n")

    interval = get_best_interval()
    left, right = interval


if __name__ == '__main__':
    do_it()
