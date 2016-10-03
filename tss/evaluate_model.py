import cPickle
import gzip
import os

import matplotlib.pyplot as plt
import numpy
import theano
from conv import get_best_interval, prepare_data, create_default_network, get_model_parameters_path, get_dataset_types

from util.data import divide_data


def get_errors(batch_size, model, test_x, test_y):
    sum_errors = 0.0
    n_errors = 0
    for i in xrange(0, test_x.shape[0] - batch_size + 1, batch_size):
        y = model.predict(test_x[i: i + batch_size])
        sum_errors += numpy.mean(numpy.not_equal(y, test_y[i: i + batch_size]).astype(float))
        n_errors += 1
    test_error = sum_errors / n_errors
    return test_error


def main():
    theano.config.openmp = True
    left, right = get_best_interval()

    batch_size = 1000

    if not os.path.exists('evaluation'):
        os.mkdir('evaluation')

    result_file = open("evaluation/results.txt", 'w')

    for data_name in get_dataset_types():
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(data_name)
        for k in xrange(3):
            model_path = get_model_parameters_path(data_name, k)
            result_file.write(model_path + "\n")
            if not os.path.exists(model_path):
                continue

            print(model_path)

            training, validation, test = prepare_data(divide_data(data_name, k), left, right)

            model = create_default_network(batch_size)

            with gzip.open(model_path, 'r') as f:
                loaded_state = cPickle.load(f)
                model.load_state(loaded_state)

            train_x, train_y = training
            valid_x, valid_y = validation
            test_x, test_y = test
            test_x = test_x.get_value()
            test_y = test_y.get_value()
            probabilities = []

            train_error = get_errors(batch_size, model, train_x.get_value(), train_y.get_value())
            result_file.write("train error: {}\n".format(train_error))
            valid_error = get_errors(batch_size, model, valid_x.get_value(), valid_y.get_value())
            result_file.write("valid error: {}\n".format(valid_error))
            test_error = get_errors(batch_size, model, test_x, test_y)
            result_file.write("test  error: {}\n".format(test_error))

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

            area_roc = 0.0
            for i in xrange(1, len((fprs))):
                dx = - fprs[i] + fprs[i - 1]
                y = (tprs[i] + tprs[i - 1]) / 2.0
                area_roc += dx * y


            result_file.write("area under ROC: {}\n".format(area_roc))
            result_file.write("\n")

        plt.savefig("evaluation/roc_{}.png".format(data_name))
        plt.clf()

    result_file.close()

if __name__ == '__main__':
    main()
