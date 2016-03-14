import os
import theano
import gzip
import cPickle

from conv import get_best_interval, prepare_data, create_default_network, get_model_parameters_path, get_dataset_types
from data import divide_data

import matplotlib.pyplot as plt


def main():
    theano.config.openmp = True
    left, right = get_best_interval()

    batch_size = 1000

    for data_name in get_dataset_types():
        for i in xrange(3):
            model_path = get_model_parameters_path(data_name, i)
            if not os.path.exists(model_path):
                continue

            print(model_path)

            training, validation, test = prepare_data(divide_data(data_name, i), interval=(left, right))

            model = create_default_network(right - left, batch_size)

            with gzip.open(model_path, 'r') as f:
                loaded_state = cPickle.load(f)
                model.load_state(loaded_state)

            test_x, test_y = test
            test_x = test_x.get_value()
            test_y = test_y.get_value()
            probabilities = []

            for i in xrange(0, test_x.shape[0] - batch_size + 1, batch_size):
                p = model.prob(test_x[i: i + batch_size])[:, 1]
                probabilities.extend(p.tolist())

            tprs = []
            fprs = []

            recalls = []
            precisions = []

            for threshold in list(sorted(probabilities)):
                table = [[0, 0], [0, 0]]
                for p, y in zip(probabilities, test_y):
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
                # print table, precision, recall, tpr, fpr

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
