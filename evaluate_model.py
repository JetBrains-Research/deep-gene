import theano
import gzip
import cPickle

from conv import get_best_interval, prepare_data, create_default_network
from data import divide_data

import matplotlib.pyplot as plt


def main():
    theano.config.openmp = True
    left, right = get_best_interval()

    batch_size = 1000

    training, validation, test = prepare_data(divide_data(0), interval=(left, right))

    data_x, data_y = training

    model = create_default_network(right - left, batch_size, data_x, data_y)

    with gzip.open('models/best_conv_model_0.pkl.gz', 'r') as f:
        loaded_state = cPickle.load(f)
        model.load_state(loaded_state)

    test_x, test_y = test
    probs = []
    for i in xrange(0, len(test_x), batch_size):
        p = model.prob(test_x[i: i + batch_size])[:, 1]
        probs.extend(p.tolist())

    tprs = []
    fprs = []

    recalls = []
    precisions = []

    for threshold in list(sorted(probs)):
        table = [[0, 0], [0, 0]]
        for p, y in zip(probs, test_y):
            pred = 1 if p > threshold else 0
            table[pred][y] += 1

        tp = float(table[1][1])
        fp = float(table[1][0])
        pos = tp + fp

        fn = float(table[0][1])
        tn = float(table[0][0])
        neg = fn + tn

        if pos == 0 or neg == 0:
            continue

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        tprs.append(tpr)
        fprs.append(fpr)

        recalls.append(recall)
        precisions.append(precision)
        print table, precision, recall, tpr, fpr

    areaROC = 0.0
    for i in xrange(1, len((fprs))):
        dx = - fprs[i] + fprs[i - 1]
        y = (tprs[i] + tprs[i - 1]) / 2.0
        areaROC += dx * y

    print("Area under ROC: {}".format(areaROC))

    areaPRC = 0.0
    for i in xrange(1, len((recalls))):
        dx = - recalls[i] + recalls[i - 1]
        y = (precisions[i] + precisions[i - 1]) / 2.0
        areaPRC += dx * y

    print("Area under PRC: {}".format(areaPRC))



if __name__ == '__main__':
    main()
