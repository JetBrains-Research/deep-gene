
import theano
import numpy
from time import gmtime, strftime
from conv import Network, prepare_data, Fitter, get_default_parameters
from data import divide_data


def test_model(data, model_params):
    training, validation, test = data
    batch_size = 1000
    rng = numpy.random.RandomState(23455)
    model = Network(rng, batch_size, model_params)

    fitter = Fitter(
        model,
        training,
        validation,
        test,
        batch_size=batch_size,
        learning_rate=model_params["learning_rate"],
        reg_coef1=model_params["reg_coef1"],
        reg_coef2=model_params["reg_coef2"])

    log_path = "logs/{}.log".format(strftime("%Y-%m-%d-%H:%M:%S", gmtime()))
    return fitter.do_fit(log_path)


def compare_different_models():
    theano.config.openmp = True
    f = open('result.txt', 'a')

    data_set = [divide_data("mm9_genes_coding", i) for i in range(0, 3)]

    for (k3) in [60, 80, 100]:
        parameters = get_default_parameters()
        parameters["n_kernels3"] = k3

        print(parameters)
        left = parameters["left"]
        right = parameters["right"]

        result = 0

        f.write("simple 3 layers\n")
        f.write(str(parameters) + "\n")

        for data in data_set:
            prepared_data = prepare_data(data, interval=(left, right))

            r = test_model(prepared_data, parameters)
            result += r / 3

            f.write("result: {}\n".format(r))
            f.flush()

        f.write("mean result: {}\n".format(result))
        f.write("\n")
        f.flush()
    f.close()


if __name__ == '__main__':
    compare_different_models()
