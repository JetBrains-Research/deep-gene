from time import gmtime, strftime

import numpy
import theano
from conv_model import TssPredictionNetwork
from conv import prepare_data, Fitter, get_default_parameters

from util.data import divide_data


def test_model(data, model_params):
    training, validation, test = data
    batch_size = 1000

    model = TssPredictionNetwork(batch_size, model_params)

    fitter = Fitter(
        model,
        training,
        validation,
        test,
        batch_size=batch_size,
        learning_rate=model_params["learning_rate"],
        L1_reg_coef=model_params["L1_reg_coef"],
        L2_reg_coef=model_params["L1_reg_coef"])

    log_path = "logs/{}.log".format(strftime("%Y-%m-%d-%H:%M:%S", gmtime()))
    return fitter.do_fit(log_path)


def compare_models_parameters():
    theano.config.openmp = True
    f = open('result.txt', 'a')

    data_set = [divide_data("mm9_cage_near_coding", i) for i in range(0, 3)]

    for k1 in [20, 30, 40]:
        parameters = get_default_parameters()
        parameters["n_kernels1"] = k1

        print(parameters)

        result = 0

        f.write("simple 3 layers\n")
        f.write(str(parameters) + "\n")

        for data in data_set:
            prepared_data = prepare_data(data, parameters)

            r = test_model(prepared_data, parameters)
            result += r / 3

            f.write("result: {}\n".format(r))
            f.flush()

        f.write("mean result: {}\n".format(result))
        f.write("\n")
        f.flush()
    f.close()


if __name__ == '__main__':
    compare_models_parameters()
