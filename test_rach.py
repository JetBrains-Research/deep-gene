import random
import theano
from conv import get_best_interval, create_default_network, Fitter, prepare_data

__author__ = 'atsky'


def load_sequence(file):
    with open(file) as f:
        pos = [line.split("\t") for line in f.readlines()]
    return pos


def combine(pos, neg):
    combined = zip(pos, [1] * len(pos)) + zip(neg, [0] * len(neg))
    random.shuffle(combined)

    return combined


def divide_data(data_type):
    def flatten_first(data):
        result = []
        for (seqs, y) in data:
            for e in seqs:
                result.append((e, y))
        random.shuffle(result)
        return result

    pos_test = load_sequence("data/{}_gh18_test.txt".format(data_type))
    neg_test = load_sequence("data/{}_gh18_test_negative.txt".format(data_type))
    pos_train = load_sequence("data/{}_gh18_train.txt".format(data_type))
    neg_train = load_sequence("data/{}_gh18_train_negative.txt".format(data_type))
    random.shuffle(pos_train)
    random.shuffle(neg_train)
    to_cut = len(pos_train) / 5
    pos_valid = pos_train[:to_cut]
    pos_train = pos_train[to_cut:]
    to_cut = len(neg_train) / 5
    neg_valid = neg_train[:to_cut]
    neg_train = neg_train[to_cut:]
    train = combine(pos_train * 20, neg_train)
    valid = combine(pos_valid * 20, neg_valid)
    test = combine(pos_test * 20, neg_test)
    train = flatten_first(train)
    valid = flatten_first(valid)
    test = flatten_first(test)
    return train, valid, test


def train_model(data, data_name):
    training, validation, test = data
    batch_size = 1000
    network = create_default_network(batch_size)
    fitter = Fitter(network,
                    training,
                    validation,
                    test,
                    batch_size=batch_size,
                    learning_rate=0.001,
                    reg_coef1=0.00001,
                    reg_coef2=0.00001)

    fitter.do_fit('evaluation/rach_{}.log'.format(data_name),
                  "evaluation/rach_{}.pkl.gz".format(data_name))


def main():
    theano.config.openmp = True
    left, right = get_best_interval()

    data = prepare_data(divide_data("np"), interval=(left, right))
    train_model(data, "np")

    data = prepare_data(divide_data("wp"), interval=(left, right))
    train_model(data, "wp")




if __name__ == '__main__':
    main()
