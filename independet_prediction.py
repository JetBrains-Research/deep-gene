import lasagne
import theano
import theano.tensor as T

from transcription_prediction import prepare_data, divide_data
from util.logs import get_result_directory_path, FileLogger


class InputTransformationLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, name=None, add_nonlinearity=True):
        super(InputTransformationLayer, self).__init__(incoming, name)

        num_inputs = self.input_shape[1]
        self.add_nonlinearity = add_nonlinearity
        self.b = self.add_param(lasagne.init.Normal(), (num_inputs, num_units), name="b")
        self.logW = self.add_param(lasagne.init.Normal(mean=-1), (num_inputs, num_units), name="logW")
        self.logC = self.add_param(lasagne.init.Normal(mean=1), (num_inputs, num_units), name="logC")

    def get_output_for(self, input, **kwargs):
        input_dimshuffle = input.dimshuffle([0, 1, 'x'])

        t = T.exp(self.logC) * T.tanh(T.exp(self.logW) * input_dimshuffle + self.b)
        if self.add_nonlinearity:
            return T.tanh(T.sum(t, axis=2))
        else:
            return T.sum(t, axis=2)

    def get_output_shape_for(self, input_shape):
        return input_shape


class Fitter(object):
    def __init__(self,
                 training,
                 validation,
                 test,
                 batch_size,
                 is_double):
        self.batch_size = batch_size
        train_set_x, train_set_s, train_set_y = training
        validation_set_x, validation_set_s, validation_set_y = validation
        test_set_x, test_set_s, test_set_y = test

        x = T.matrix('x')
        y = T.vector("y")

        self.x = x
        self.y = y

        index = T.lscalar()  # index to a [mini]batch

        if is_double:
            output_layer, l_is_coding = self.create_double_chip_seq_regression(x)
            regression = lasagne.layers.get_output(output_layer)
            is_coding = lasagne.layers.get_output(l_is_coding)
            output = (regression[:, 0] * (1 - is_coding) + regression[:, 1] * is_coding)
        else:
            output_layer = self.create_chip_seq_regression(x)
            output = lasagne.layers.get_output(output_layer).flatten()

        err = T.mean(lasagne.objectives.squared_error(output, y))

        l1_penalty = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l1)
        l2_penalty = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l2)

        cost = err + l1_penalty * 1e-4 + l2_penalty * 1e-4

        params = lasagne.layers.get_all_params(output_layer, trainable=True)

        updates = lasagne.updates.adam(cost, params, learning_rate=0.005)

        self.train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                y: train_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        output_deterministic = lasagne.layers.get_output(output_layer, deterministic=True).flatten()

        err = T.mean(lasagne.objectives.squared_error(output_deterministic, y))

        self.get_validation_error = theano.function(
            inputs=[index],
            outputs=err,
            givens={
                x: validation_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                y: validation_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

        self.get_test_error = theano.function(
            inputs=[index],
            outputs=err,
            givens={
                x: test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                y: test_set_y[index * self.batch_size:(index + 1) * self.batch_size]
            }
        )

    def create_chip_seq_regression(self, x):
        input = lasagne.layers.InputLayer(shape=(None, 89), input_var=x)
        transformed = InputTransformationLayer(input, 5)
        regression = lasagne.layers.DenseLayer(transformed, 1, nonlinearity=None)

        return regression

    def create_double_chip_seq_regression(self, x):
        l_input = lasagne.layers.InputLayer(shape=(None, 89), input_var=x)

        l_is_coding = lasagne.layers.SliceLayer(l_input, indices=0, axis=1)
        l_chip = lasagne.layers.SliceLayer(l_input, indices=slice(1, 89), axis=1)

        transformed = InputTransformationLayer(l_chip, 4)
        l_regression = lasagne.layers.DenseLayer(transformed, 2, nonlinearity=None)

        return l_regression, l_is_coding


def get_validation_error(fitter):
    valid_err = 0.0
    for i in range(5):
        valid_err += fitter.get_validation_error(i) / 5
    return valid_err


def get_test_error(fitter):
    test_err = 0.0
    for i in range(5):
        test_err += fitter.get_test_error(i) / 5
    return test_err


def get_error_from_seq(data, logger, is_double):
    train, validation, test = data
    train_x, train_s, train_y = train
    batch_size = 1000
    train_batches_number = train_x.get_value().shape[0] // batch_size

    fitter = Fitter(train, validation, test, batch_size, is_double)
    best_error = 1000
    patience = 100
    result_error = 0
    for epoch in range(1000):
        err = 0.0

        for i in range(train_batches_number):
            err += fitter.train_model(i)

        valid_err = get_validation_error(fitter)

        logger.log("{:3} total error: {:.3f} valid error {:.3f} patience: {}".format(
            epoch,
            err / train_batches_number,
            valid_err,
            patience))

        if valid_err < best_error:
            best_error = valid_err
            test_err = get_test_error(fitter)
            result_error = test_err
            logger.log("      valid_err: {}".format(valid_err))
            logger.log("       test_err: {}".format(test_err))
            patience = 100

        else:
            patience -= 1
            if patience == 0: break

    logger.log(result_error)
    return result_error


def main():
    theano.config.openmp = True
    # theano.config.optimizer = "None"

    result_directory = get_result_directory_path("independent_prediction")

    logger = FileLogger(result_directory, "results")

    logger.log("Dependent")

    for i in range(5):
        data = prepare_data(divide_data("CD4", i + 1), 1000, 2500)

        fitter_logger = FileLogger(result_directory, "log_{}".format(i))
        error = get_error_from_seq(data, fitter_logger, True)
        fitter_logger.close()
        logger.log("error: {}".format(error))

    logger.log("Independent")

    for i in range(5):
        data = prepare_data(divide_data("CD4", i + 1), 1000, 2500)

        fitter_logger = FileLogger(result_directory, "log_{}".format(i))
        error = get_error_from_seq(data, fitter_logger, False)
        fitter_logger.close()
        logger.log("error: {}".format(error))

    logger.close()


if __name__ == '__main__':
    main()
