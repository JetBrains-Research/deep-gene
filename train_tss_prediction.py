from tss.conv import *
from tss.conv_model import TssPredictionNetwork
from util.data import divide_data


def train_model(data, dataset_name, index):
    training, validation, test = data
    batch_size = 1000
    parameters = get_default_parameters()
    network = TssPredictionNetwork(batch_size, parameters)
    fitter = Fitter(network,
                    training,
                    validation,
                    test,
                    batch_size=batch_size,
                    learning_rate=parameters["learning_rate"],
                    L1_reg_coef=parameters["L1_reg_coef"],
                    L2_reg_coef=parameters["L1_reg_coef"],)
    model_name = get_model_name(dataset_name, index)
    log_path = 'models/{}.log'.format(model_name)
    model_path = 'models/{}.pkl.gz'.format(model_name)
    fitter.do_fit(log_path, model_path)


def main():
    theano.config.openmp = True

    for data_name in get_dataset_types():
        for i in xrange(3):
            path = get_model_parameters_path(data_name, i)
            if os.path.exists(path):
                print "Model {} exists.".format(path)
                continue
            data_set = divide_data(data_name, i)

            default_parameters = get_default_parameters()
            data = prepare_data(data_set, default_parameters)
            train_model(data, data_name, index=i)


if __name__ == '__main__':
    main()
