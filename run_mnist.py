"""Main function to run the code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import argparse
from src.data_provider import datasets_factory
from src.models.model_factory import Model
import src.trainer as trainer
from src.utils import preprocess
from src.utils import file_util
import paddle.fluid as fluid

# -----------------------------------------------------------------------------
def str_to_bool(str):
	return True if str.lower() == 'true' else False

parser = argparse.ArgumentParser()

parser.add_argument("--train_data_paths", type=str, default="/home/aistudio/data/data18733/mnist/moving-mnist-train.npz", help="train data path.")
parser.add_argument("--valid_data_paths", type=str, default="/home/aistudio/data/data18733/mnist/moving-mnist-valid.npz", help="validation data paths.")
parser.add_argument("--save_dir", type=str, default="checkpoints/_mnist_e3d_lstm", help="dir to store trained net.")
parser.add_argument("--gen_frm_dir", type=str, default="results/_mnist_e3d_lstm", help="dir to store result.")

parser.add_argument("--is_training", type=str_to_bool, default="True", help="training or testing.")
parser.add_argument("--dataset_name", type=str, default="mnist", help="The name of dataset.")
parser.add_argument("--input_length", type=int, default=10, help="input length.")
parser.add_argument("--total_length", type=int, default=20, help="total input and output length.")
parser.add_argument("--img_width", type=int, default=64, help="input image width.")
parser.add_argument("--img_channel", type=int, default=1, help="number of image channel.")
parser.add_argument("--patch_size", type=int, default=4, help="patch size on one dimension.")
parser.add_argument("--reverse_input", type=str_to_bool, default="False", help="reverse the input/outputs during training.")

parser.add_argument("--model_name", type=str, default="gru_3d", help="The name of the architecture.")
parser.add_argument("--pretrained_model", type=str, default="", help="model file to initialize from.")
parser.add_argument("--num_hidden", type=str, default="64,64,64,64", help="COMMA separated number of units of e3d lstms.")
parser.add_argument("--filter_size", type=int, default=5, help="filter of a e3d lstm layer.")
parser.add_argument("--layer_norm", type=str_to_bool, default="True", help="whether to apply tensor layer norm.")

parser.add_argument("--scheduled_sampling", type=str_to_bool, default="True", help="for scheduled sampling.")
parser.add_argument("--sampling_stop_iter", type=int, default=50000 , help="for scheduled sampling.")
parser.add_argument("--sampling_start_value", type=float, default=1.0, help="for scheduled sampling.")
parser.add_argument("--sampling_changing_rate", type=float, default=0.00002 , help="for scheduled sampling.")

parser.add_argument("--lr", type=float, default=0.001, help="learning rate.")
parser.add_argument("--batch_size", type=int, default=8, help="batch size for training.")
parser.add_argument("--max_iterations", type=int, default=10000000, help="max num of steps.")
parser.add_argument("--display_interval", type=int, default=1, help="number of iters showing training loss.")
parser.add_argument("--test_interval", type=int, default=1000, help="number of iters for test.")
parser.add_argument("--snapshot_interval", type=int, default=500, help="number of iters saving models.")
parser.add_argument("--num_save_samples", type=int, default=10, help="number of sequences to be saved.")
parser.add_argument("--n_gpu", type=int, default=1, help="how many GPUs to distribute the training across.")
parser.add_argument("--allow_gpu_growth", type=str_to_bool, default="True", help="allow gpu growth.")

configs, _  = parser.parse_known_args()


def main():
    """Main function."""

    if not file_util.Exists(configs.save_dir):
        file_util.MakeDirs(configs.save_dir)
    if file_util.Exists(configs.gen_frm_dir):
        file_util.DeleteRecursively(configs.gen_frm_dir)
    file_util.MakeDirs(configs.gen_frm_dir)

    print('Initializing models')

    model = Model(configs)

    if configs.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)


def schedule_sampling(eta, itr):
    """Gets schedule sampling parameters for training."""
    zeros = np.zeros(
        (configs.batch_size, configs.total_length - configs.input_length - 1,
        configs.img_width // configs.patch_size, configs.img_width // configs.patch_size,
        configs.patch_size**2 * configs.img_channel), dtype=np.float32)
    if not configs.scheduled_sampling:
        return 0.0, zeros

    if itr < configs.sampling_stop_iter:
        eta -= configs.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (configs.batch_size, configs.total_length - configs.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones(
        (configs.img_width // configs.patch_size, configs.img_width // configs.patch_size,
        configs.patch_size**2 * configs.img_channel), dtype=np.float32)
    zeros = np.zeros(
        (configs.img_width // configs.patch_size, configs.img_width // configs.patch_size,
        configs.patch_size**2 * configs.img_channel), dtype=np.float32)
    real_input_flag = []
    for i in range(configs.batch_size):
        for j in range(configs.total_length - configs.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(
        real_input_flag,
        (configs.batch_size, configs.total_length - configs.input_length - 1,
        configs.img_width // configs.patch_size, configs.img_width // configs.patch_size,
        configs.patch_size**2 * configs.img_channel))
    return eta, real_input_flag


def train_wrapper(model):
    """Wrapping function to train the model."""
    if configs.pretrained_model:
        model.load(configs.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        configs.dataset_name,
        configs.train_data_paths,
        configs.valid_data_paths,
        configs.batch_size * configs.n_gpu,
        configs.img_width,
        seq_length=configs.total_length,
        is_training=True)

    eta = configs.sampling_start_value

    for itr in range(1, configs.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        if configs.dataset_name == 'penn':
            ims = ims['frame']
        if ims is None:
            continue
        ims = preprocess.reshape_patch(ims, configs.patch_size)

        eta, real_input_flag = schedule_sampling(eta, itr)

        trainer.train(model, ims, real_input_flag, configs, itr)

        if itr % configs.snapshot_interval == 0:
            model.save(itr)

        if itr % configs.test_interval == 0:
            trainer.test(model, test_input_handle, configs, itr)

        train_input_handle.next()


def test_wrapper(model):
    model.load(configs.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        configs.dataset_name,
        configs.train_data_paths,
        configs.valid_data_paths,
        configs.batch_size * configs.n_gpu,
        configs.img_width,
		seq_length=configs.total_length,
        is_training=False)
    trainer.test(model, test_input_handle, configs, 'test_result')


if __name__ == '__main__':
    main()
