import theano
from conv import get_best_interval

from transcription_prediction import prepare_data, divide_data, get_error_from_seq


def update_mask(base_mask):
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
            data = prepare_data(divide_data(get_best_interval(), mask=mask))
            error += get_error_from_seq("chip-seq", data) / 5

        print "mask"
        print mask
        print "error: {}".format(error)
        errors.append(error)
        print "errors: {}".format(errors)

        if error < best_error:
            best_error = error
            best_mask = list(mask)
    return best_mask, best_error


def optimize_mask():
    masks = []
    errors = []

    mask = [0.0] * 176
    for i in range(50):
        mask, error = update_mask(mask)
        masks.append(list(mask))
        errors.append(error)

        with open("masks.txt", 'w') as f:
            for m, e in zip(masks, errors):
                f.write(str(m) + "\n")
                f.write(str(e) + "\n")


def main():
    theano.config.openmp = True

    optimize_mask()


if __name__ == '__main__':
    main()
