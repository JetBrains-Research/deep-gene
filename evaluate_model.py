import os
import shutil
import numpy
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

    if not os.path.exists('evaluation'):
        os.mkdir('evaluation')

    result_file = open("evaluation/results.txt", 'w')

    for data_name in get_dataset_types():
        for k in xrange(3):
            model_path = get_model_parameters_path(data_name, k)
            result_file.write(model_path + "\n")
            if not os.path.exists(model_path):
                continue

            print(model_path)

            training, validation, test = prepare_data(divide_data(data_name, k), interval=(left, right))

            model = create_default_network(batch_size)

            with gzip.open(model_path, 'r') as f:
                loaded_state = cPickle.load(f)
                model.load_state(loaded_state)

            test_x, test_y = test
            test_x = test_x.get_value()
            test_y = test_y.get_value()
            probabilities = []

            sum_errors = 0.0
            n_errors = 0

            for i in xrange(0, test_x.shape[0] - batch_size + 1, batch_size):
                y = model.predict(test_x[i: i + batch_size])
                sum_errors += numpy.mean(numpy.not_equal(y, test_y[i: i + batch_size]).astype(float))
                n_errors += 1

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

            plt.plot(fprs, tprs)
            plt.savefig("evaluation/roc_{}_{}.png".format(data_name, k))

            with open("evaluation/table_{}_{}.csv".format(data_name, k), 'w') as f:
                for tpr, fpr in zip(tprs, fprs):
                    f.write("{}\t{}\n".format(tpr, fpr))

            area_roc = 0.0
            for i in xrange(1, len((fprs))):
                dx = - fprs[i] + fprs[i - 1]
                y = (tprs[i] + tprs[i - 1]) / 2.0
                area_roc += dx * y

            result_file.write("mean error: {}\n".format(sum_errors / n_errors))
            result_file.write("area under ROC: {}\n".format(area_roc))
            result_file.write("\n")

    result_file.close()

if __name__ == '__main__':
    main()
