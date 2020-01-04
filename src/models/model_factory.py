"""Factory to get E3D-LSTM models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from src.models import gru_3d_net
import paddle.fluid as fluid
import numpy as np

def get_params(program, generator=True):
	all_params=program.global_block().all_parameters()
	if generator:
	    return [t.name for t in all_params if not t.name.startswith("dis")]
	else:
	    return [t.name for t in all_params if t.name.startswith("dis")]



class Model(object):
    """Model class for E3D-LSTM model."""

    def __init__(self, configs):

        self.configs = configs

        num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        num_layers = len(num_hidden)

        self.discriminator_prog = fluid.Program()
        self.generator_prog = fluid.Program()
        startup = fluid.Program()

        # with fluid.program_guard(self.discriminator_prog, startup):
        #     x = fluid.data(shape=[
        #             self.configs.batch_size,
        #             self.configs.total_length,
        #             self.configs.img_width // self.configs.patch_size,
        #             self.configs.img_width // self.configs.patch_size,
        #             self.configs.patch_size * self.configs.patch_size * self.configs.img_channel
        #         ], dtype='float32', name='x') #(2, 30, 16, 16, 64)

        #     real_input_flag = fluid.data(shape=[
        #         self.configs.batch_size,
        #         self.configs.total_length - self.configs.input_length - 1,
        #         self.configs.img_width // self.configs.patch_size,
        #         self.configs.img_width // self.configs.patch_size,
        #         self.configs.patch_size * self.configs.patch_size *
        #         self.configs.img_channel
        #     ], dtype='float32', name='real_input_flag')

        #     # define a model
        #     output_list = self.construct_model(x, real_input_flag, num_layers, num_hidden)

        #     loss_d = output_list[2]

        #     self.train_op_d = [loss_d]
        #     d_params = get_params(self.discriminator_prog, False)
        #     adam_optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
        #     adam_optimizer.minimize(loss_d, parameter_list=d_params)

        with fluid.program_guard(self.generator_prog, startup):
            # f1 = fluid.data(name="f1", shape=[1], dtype="float32")
            # f2 = fluid.data(name="f2", shape=[1], dtype="float32")
            x = fluid.data(shape=[
                    self.configs.batch_size,
                    self.configs.total_length,
                    self.configs.img_width // self.configs.patch_size,
                    self.configs.img_width // self.configs.patch_size,
                    self.configs.patch_size * self.configs.patch_size * self.configs.img_channel
                ], dtype='float32', name='x') #(2, 30, 16, 16, 64)

            real_input_flag = fluid.data(shape=[
                self.configs.batch_size,
                self.configs.total_length - self.configs.input_length - 1,
                self.configs.img_width // self.configs.patch_size,
                self.configs.img_width // self.configs.patch_size,
                self.configs.patch_size * self.configs.patch_size *
                self.configs.img_channel
            ], dtype='float32', name='real_input_flag')

            # define a model
            output_list = self.construct_model(x, real_input_flag, num_layers, num_hidden)

            gen_ims = output_list[0]
            loss = output_list[1]
            # g_loss = output_list[3]

            self.pred_seq = gen_ims
            # loss_g = loss * f1 + g_loss * f2
            loss_g = loss
            self.train_op_g = [loss_g]

            self.test_program = self.generator_prog.clone(for_test=True)

            g_params = get_params(self.generator_prog, True)
            adam_optimizer = fluid.optimizer.Adam(learning_rate=configs.lr)
            adam_optimizer.minimize(loss_g, parameter_list=g_params)

        self.test_op = [self.pred_seq]

        place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        self.exe.run(startup)


    def train(self, inputs, real_input_flag, itr):
        feed_dict = {}
        feed_dict.update({'x': inputs})
        feed_dict.update({'real_input_flag': real_input_flag})
        # if itr % 1000 < 800:
        #     # 正常训练
        #     feed_dict.update({'f1': np.array([1]).astype(np.float32)})
        #     feed_dict.update({'f2': np.array([0]).astype(np.float32)})
        #     loss = self.exe.run(program=self.generator_prog, feed=feed_dict, fetch_list=self.train_op_g)
        # else:
        #     # 对抗训练
        #     offset = itr % 1000
        #     if offset % 5 == 1:
        #         loss = self.exe.run(program=self.discriminator_prog, feed=feed_dict, fetch_list=self.train_op_d)
        #     else:
        #         feed_dict.update({'f1': np.array([0]).astype(np.float32)})
        #         feed_dict.update({'f2': np.array([1]).astype(np.float32)})
        #         loss = self.exe.run(program=self.generator_prog, feed=feed_dict, fetch_list=self.train_op_g)
        loss = self.exe.run(program=self.generator_prog, feed=feed_dict, fetch_list=self.train_op_g)
        return loss

    def test(self, inputs, real_input_flag):
        feed_dict = {}
        feed_dict.update({'x': inputs})
        feed_dict.update({'real_input_flag': real_input_flag})
        # feed_dict.update({'f1': np.array([1]).astype(np.float32)})
        # feed_dict.update({'f2': np.array([0]).astype(np.float32)})
        gen_ims = self.exe.run(program=self.test_program, feed=feed_dict, fetch_list=self.test_op)
        return gen_ims

    def save(self, itr):
        prog = self.discriminator_prog
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_d')
        fluid.save(prog, checkpoint_path)

        prog = self.generator_prog
        checkpoint_path = os.path.join(self.configs.save_dir, 'model_g')
        fluid.save(prog, checkpoint_path)
        print('saved to ' + self.configs.save_dir)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        prog = self.discriminator_prog
        fluid.load(prog, checkpoint_path + "_d")
        prog = self.generator_prog
        fluid.load(prog, checkpoint_path + "_g")

    def construct_model(self, images, real_input_flag, num_layers, num_hidden):
        """Contructs a model."""
        networks_map = {
            'gru_3d': gru_3d_net.rnn,
        }

        if self.configs.model_name in networks_map:
            func = networks_map[self.configs.model_name]
            return func(images, real_input_flag, num_layers, num_hidden, self.configs)#
        else:
            raise ValueError('Name of network unknown %s' % self.configs.model_name)
