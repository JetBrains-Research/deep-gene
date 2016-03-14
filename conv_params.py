
import theano
import numpy
from time import gmtime, strftime
from conv import Network, prepare_data, Fitter
from data import divide_data


def test_model(data, model_params, sequence_size=2000):
    training, validation, test = data
    batch_size = 1000
    rng = numpy.random.RandomState(23455)
    model = Network(rng,
                    batch_size,
                    n_kernels1=model_params["n_kernels1"],
                    n_kernels2=model_params["n_kernels2"],
                    n_kernels3=model_params["n_kernels3"],
                    pattern1_size=model_params["pattern1_size"],
                    pattern2_size=model_params["pattern2_size"],
                    pattern3_size=model_params["pattern3_size"],
                    pool2_size=model_params["pool2_size"],
                    sequence_size=sequence_size)

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

    data_set = [divide_data("genes-coding", i) for i in range(0, 3)]

    for learning_rate in [0.003, 0.001, 0.0003]:
        current_parameters = {
            "left": 1000,
            "right": 2500,
            "learning_rate": learning_rate,
            "n_kernels1": 32,
            "n_kernels2": 64,
            "n_kernels3": 64,
            "pattern1_size": 4,
            "pattern2_size": 6,
            "pattern3_size": 6,
            "pool2_size": 2,
            "reg_coef1": 0.00001,
            "reg_coef2": 0.00001,
        }

        print(current_parameters)
        left = current_parameters["left"]
        right = current_parameters["right"]

        result = 0

        f.write("simple 3 layers\n")
        f.write(str(current_parameters) + "\n")

        for data in data_set:
            prepared_data = prepare_data(data, interval=(left, right))

            r = test_model(prepared_data, current_parameters, sequence_size=right - left)
            result += r / 3

            f.write("result: {}\n".format(r))
            f.flush()

        f.write("mean result: {}\n".format(result))
        f.write("\n")
        f.flush()
    f.close()


if __name__ == '__main__':
    compare_different_models()
