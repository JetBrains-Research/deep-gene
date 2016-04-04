import gzip
import cPickle
import dominate
from dominate.tags import *
import numpy
import theano
import theano.tensor as T
from conv import get_best_interval, prepare_data, get_model_parameters_path, create_default_network
from data import divide_data


class Relevance(object):
    def __init__(self,
                 network):
        x = network.x
        conv_input = network.conv_input
        is_train = network.is_train

        grads = T.grad(network.regression.p_y_given_x[0][1], conv_input)

        self.get_grad = theano.function(
            [x],
            grads,
            givens={
                is_train: numpy.cast['int32'](0)
            }
        )


def get_color(alpha):
    return (numpy.array([255.0, 255.0, 255.0]) * (1 - alpha) + numpy.array([255.0, 0.0, 0.0]) * alpha).tolist()


def as_hex(color):
    def to_hex(x):
        return "{:x}".format(int(x)).zfill(2)

    return "".join(map(to_hex, color))


def add_table(doc, result, x):
    left, right = get_best_interval()
    max_relevance = max(result)
    t = table(border=0, cellspacing=0)
    doc.add(t)
    with t.add(tbody()):
        with tr():
            for offset in range(left, right, 10):
                td(offset - 2000,
                   style="font-family: monospace; text-align:left;",
                   colspan=10)

        l = tr()

        for ch, r in zip(x[left:right], result):
            cell = td(ch)
            color = as_hex(get_color(r / max_relevance))
            cell["style"] = "background-color: #{}; font-family: monospace; text-align:center;".format(color)
            l += cell


def main():
    theano.config.openmp = True
    batch_size = 1
    left, right = get_best_interval()

    data_name, i = "mm9_genes_coding", 0

    data_set = divide_data(data_name, i)

    data = prepare_data(data_set, interval=(left, right))
    sequences = data[0][0].get_value()

    network = create_default_network(batch_size)

    with gzip.open(get_model_parameters_path(data_name, i), 'r') as f:
        loaded_state = cPickle.load(f)
        network.load_state(loaded_state)
    relevance = Relevance(network)

    doc = dominate.document(title='Tss prediction report')

    for i in range(2000):
        x, y = data_set[0][i]
        prediction = network.prob(sequences[i:i + 1])[0][1]

        if not (y == 0 and prediction > 0.7):
            continue

        doc += div("gene: {}".format(y))
        doc += div("gene probability: {}".format(prediction))

        grad = relevance.get_grad(sequences[i:i + 1])[0, :, :, 0]

        result = numpy.sqrt(numpy.sum((grad ** 2), 0)).tolist()

        add_table(doc, result, x)

    with open('report.html', 'w') as f:
        f.write(doc.render())


if __name__ == '__main__':
    main()
