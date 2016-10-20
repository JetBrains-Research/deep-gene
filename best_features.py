import theano

from transcription_prediction import prepare_data, divide_data, get_error_from_seq
from util.logs import PrintLogger, get_result_directory_path, FileLogger


def update_mask(base_mask, data_list, logger):
    errors = []
    best_error = float("inf")
    best_mask = None

    for j in range(176):
        if base_mask[j] == 1.0:
            continue

        mask = list(base_mask)

        mask[j] = 1.0

        error = 0

        for i in range(5):
            print "start: {}".format(i)
            data = prepare_data(data_list[i], 1000, 2500, mask)

            error += get_error_from_seq("chip-seq", data, PrintLogger()) / 5

        logger.log("mask")
        logger.log(mask)
        logger.log("error: {}".format(error))
        errors.append(error)
        logger.log("errors: {}".format(errors))

        if error < best_error:
            best_error = error
            best_mask = list(mask)
    return best_mask, best_error


def optimize_mask(result_directory, data_list):
    masks = []
    errors = []

    logger = FileLogger(result_directory, "main")

    mask = [0.0] * 176
    for i in range(50):
        mask, error = update_mask(mask, data_list, logger)
        masks.append(list(mask))
        errors.append(error)

        with open("masks.txt", 'w') as f:
            for m, e in zip(masks, errors):
                f.write(str(m) + "\n")
                f.write(str(e) + "\n")

    logger.close()


def main():
    theano.config.openmp = True

    data_list = [divide_data("CD4", i + 1) for i in range(5)]

    result_directory = get_result_directory_path("transcription_best_features")
    optimize_mask(result_directory, data_list)


if __name__ == '__main__':
    main()
