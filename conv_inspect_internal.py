import gzip
import cPickle

from conv import get_best_interval, create_network, prepare_data
from data import divide_data


def get_all_layers(data_x, model, i=0):
    (conv_1_output, conv_2_output, conv_3_output, mr_layer_output, fully_connected_output,
     p_y_given_x) = model.get_layers(data_x[i:i+1000])
    return {
        "conv_1": conv_1_output[0, :, :, 0],
        "conv_2": conv_2_output[0, :, :, 0],
        "conv_3": conv_3_output[0, :, :, 0],
        "mr_layer": mr_layer_output[0],
        "fully_connected": fully_connected_output[0],
        "p_y_given_x": p_y_given_x[0],}


def do_it():
    batch_size = 1000
    interval = get_best_interval()
    left, right = interval

    model = create_network(right - left, batch_size)

    with gzip.open('models/best_conv_model_genes-coding_0.pkl.gz', 'r') as f:
        model.load_state(cPickle.load(f))

    data_set = divide_data("genes-coding", 0)
    raw_data, data_x, data_y = prepare_data(data_set, interval=(left, right))

    print raw_data[0][0]

    get_all_layers(data_x, model)


if __name__ == '__main__':
    do_it()
